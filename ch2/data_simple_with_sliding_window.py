import tiktoken

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