import json
from pathlib import Path
import torch
import tensorflow as tf
import tiktoken
import time
import matplotlib.pyplot as plt

from ch4.gpt_model import GPTModel, generate_text_simple
from ch5.evaluating_model import text_to_token_ids, token_ids_to_text
from ch5.get_device import get_device
from ch5.gpt_download import load_gpt2_params_from_tf_ckpt
from ch5.load_weights_from_openai import load_weights_into_gpt
from ch6.prepare_dataset import create_data_loaders

def calc_accuracy_loader(data_loader, model ,device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i<num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:,-1,:]
            predicted_labels = logits.argmax(dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:,-1,:]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader)==0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

def evaluate_model(model, train_loader, valid_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        valid_loss = calc_loss_loader(
            valid_loader, model, device, num_batches=eval_iter
        )
    model.train()

    return train_loss, valid_loss

def train_classifier_simple(model, train_loader, validation_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, validation_losses, train_accuracies, validation_accuracies = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, validation_loss = evaluate_model(model, train_loader, validation_loader, device, eval_iter)
                train_losses.append(train_loss)
                validation_losses.append(validation_loss)
                print(f"Ep {epoch+1} step {global_step:06d}"
                      f"Train loss: {train_loss:.4f} "
                      f"Validation loss: {validation_loss:.4f}"
                      )

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        validation_accuracy = calc_accuracy_loader(
            validation_loader, model, device, num_batches=eval_iter
        )
        print(f"Train accuracy: {100*train_accuracy:.4f} ", end="")
        print(f"Validation accuracy: {100*validation_accuracy:.4f} ")
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

    return train_losses, validation_losses, train_accuracies, validation_accuracies, examples_seen

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    plt.show()

if __name__ == "__main__":
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    gpt = GPTModel(BASE_CONFIG)
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    model_size = CHOOSE_MODEL.split("(")[-1].rstrip(")")
    model_dir = Path(__file__).resolve().parents[1] / "ch5" / "gpt2" / model_size
    settings = json.loads((model_dir / "hparams.json").read_text(encoding="utf-8"))
    checkpoint = tf.train.latest_checkpoint(str(model_dir))
    params = load_gpt2_params_from_tf_ckpt(checkpoint, settings)
    load_weights_into_gpt(gpt, params)
    device = get_device()

    # text_1 = "Every effort moves you"
    # token_ids = generate_text_simple(
    #     model=gpt,
    #     idx=text_to_token_ids(text_1, tokenizer).to(device),
    #     max_new_tokens=15,
    #     context_size=BASE_CONFIG["context_length"],
    # )
    # print(token_ids_to_text(token_ids, tokenizer))

    # text_2 = (
    #     "Is the following text 'spam'? Answer with 'yes' or 'no':"
    #     " 'You are a winner you have been specially"
    #     " selected to receive $1000 cash or a $2000 award.'"
    # )
    # token_ids = generate_text_simple(
    #     model=gpt,
    #     idx=text_to_token_ids(text_2, tokenizer).to(device),
    #     max_new_tokens=23,
    #     context_size=BASE_CONFIG["context_length"]
    # )
    # print(token_ids_to_text(token_ids, tokenizer))

    # 冻结模型
    for param in gpt.parameters():
        param.requires_grad = False

    torch.manual_seed(123)
    num_classes = 2
    gpt.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"], out_features=num_classes
    )

    for param in gpt.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in gpt.final_norm.parameters():
        param.requires_grad = True
    gpt.to(device)

    torch.manual_seed(123)
    train_loader, validation_loader, test_loader = create_data_loaders(data_dir=Path(__file__).resolve().parent, batch_size=10)
    train_accuracy = calc_accuracy_loader(train_loader, gpt, device, num_batches=5)
    validation_accuracy = calc_accuracy_loader(validation_loader, gpt, device, num_batches=5)
    test_accuracy = calc_accuracy_loader(test_loader, gpt, device, num_batches=5)

    print(f"Train Accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation Accuracy: {validation_accuracy*100:.2f}%")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, gpt, device, num_batches=5)
        validation_loss = calc_loss_loader(validation_loader, gpt, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, gpt, device, num_batches=5)
    print(f"Train Loss: {train_loss:.3f}")
    print(f"Validation Loss: {validation_loss:.3f}")
    print(f"Test Loss: {test_loss:.3f}")

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    train_losses, validation_losses, train_accuracies, validation_accuracies, examples_seen = train_classifier_simple(
        gpt, train_loader, validation_loader, optimizer, device, num_epochs,
        eval_freq=50, eval_iter=5
    )

    end_time = time.time()
    exec_time = end_time - start_time
    print(exec_time)

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    plot_values(epochs_tensor, examples_seen_tensor, train_losses, validation_losses)