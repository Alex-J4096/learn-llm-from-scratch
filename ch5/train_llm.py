import tiktoken
import torch
from pathlib import Path
from ch5.evaluating_model import calc_loss_loader, calc_loss_batch
from ch5.evaluating_model import text_to_token_ids, token_ids_to_text, generate_text_simple
from ch5.get_device import get_device
from ch4.gpt_model import GPTModel
from ch4.gpt_cfg import GPT_CONFIG_124M
from ch2.data_loader import create_dataloader_v1

def train_model_simple(model, train_loader, valid_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, valid_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 计算损失梯度
            loss.backward()
            # 用损失梯度更新模型权重
            optimizer.step()
            # 从训练开始到当前时刻，模型累计处理了多少个输入 Token
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, valid_loss = evaluate_model(
                    model, train_loader, valid_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)

                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch+1} Step {global_step:06d}:"
                      f"Train loss {train_loss:.4f} Valid loss {valid_loss:.4f}"
                      )
        generate_and_print_simple(model, tokenizer, device, start_context)

    return train_losses, valid_losses, track_tokens_seen



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

def generate_and_print_simple(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    print("新生成的 token 数", token_ids.shape[1] - encoded.shape[1])
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    # device = get_device()
    device = torch.device("cpu")
    GPT_CONFIG_124M = GPT_CONFIG_124M.copy()
    GPT_CONFIG_124M["context_length"] = 256

    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1,
    )
    num_epochs = 10

    file_path = Path(__file__).resolve().parents[1] / "ch2" / "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    valid_data = text_data[split_idx:]
    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_len=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = create_dataloader_v1(
        valid_data,
        batch_size=2,
        max_len=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    train_losses, valid_losses, tokens_seen = train_model_simple(
        model, train_loader, valid_loader, optimizer, device, num_epochs=num_epochs,
        eval_freq=5, eval_iter=5, start_context="Every effort moves you",
        tokenizer=tokenizer
    )
