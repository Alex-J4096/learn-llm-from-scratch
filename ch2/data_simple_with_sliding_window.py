import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_simple = enc_text[50:]

context_size = 4
x = enc_simple[:context_size]
y = enc_simple[1:context_size+1]
# print(f"x:{x}")
# print(f"y:      {y}")

for i in range(1, context_size+1):
    context = enc_simple[:i]
    desired = enc_simple[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids)-max_len, stride):
            input_chunk = token_ids[i:i+max_len]
            target_chunk = token_ids[i+1:i+max_len+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

def create_dataloader_v1(txt, batch_size=4, max_len=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_len, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# dataloader = create_dataloader_v1(raw_text, batch_size=1, max_len=4, stride=1, shuffle=False)
# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)

# second_batch = next(data_iter)
# print(second_batch)

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_len=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print(inputs)
print(targets)
