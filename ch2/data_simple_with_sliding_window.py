import tiktoken
from data_loader import create_dataloader_v1

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
