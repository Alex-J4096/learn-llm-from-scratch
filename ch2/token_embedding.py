import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# input_ids = torch.tensor([2,3,5,1])
# vocab_size = 6
# output_dim = 3

torch.manual_seed(123)
# embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# print(embedding_layer.weight)

# print(embedding_layer(torch.tensor([3])))

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

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_len = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_len=max_len,
    stride=max_len, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token ids:\n", inputs)
print("\n inputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("\n",token_embeddings.shape)

# 获取绝对位置嵌入
context_len = max_len
pos_embedding_layer = torch.nn.Embedding(context_len, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_len))
# print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print("\n input_embeddings shape:\n", input_embeddings.shape)