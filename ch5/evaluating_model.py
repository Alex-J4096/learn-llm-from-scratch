from pathlib import Path

import torch
import tiktoken
from ch4.gpt_cfg import GPT_CONFIG_124M
from ch4.gpt_model import GPTModel
from ch4.gpt_model import generate_text_simple
from ch2.data_loader import create_dataloader_v1
from ch5.get_device import get_device

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    # 移除batch维度
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# 计算加载器加载的训练集和验证集返回的给定批次的交叉熵损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

# 计算训练集和验证集损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    # 若没指定便利多少批次则默认遍历所有批次
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

if __name__=='__main__':
    torch.manual_seed(123)
    GPT_CONFIG_124M = GPT_CONFIG_124M.copy()
    GPT_CONFIG_124M["context_length"] = 256
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    # start_context = "Every effort moves you"
    #
    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=text_to_token_ids(start_context, tokenizer),
    #     max_new_tokens=10,
    #     context_size=GPT_CONFIG_124M["context_length"],
    # )
    # print("output text:\n", token_ids_to_text(token_ids, tokenizer))

    file_path = Path(__file__).resolve().parents[1] / "ch2" / "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("total characters:", total_characters)
    print("total tokens:", total_tokens)

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

    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
    print("\nValid loader:")
    for x, y in valid_loader:
        print(x.shape, y.shape)

    device = get_device()

    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        valid_loss = calc_loss_loader(valid_loader, model, device)

    print("Train loss:", train_loss)
    print("Valid loss:", valid_loss)
