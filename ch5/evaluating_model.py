from pathlib import Path

import torch
import tiktoken
from ch4.gpt_cfg import GPT_CONFIG_124M
from ch4.gpt_model import GPTModel
from ch4.gpt_model import generate_text_simple



def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    # 移除batch维度
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

if __name__=='__main__':
    torch.manual_seed(123)
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
