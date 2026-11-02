import re

from ch2.tokenizer import SimpleTokenizerV1, SimpleTokenizerV2

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("total number of character:", len(raw_text))
# print(raw_text[:99])

# 分词
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
# print(preprocessed[:30])

# 创建词汇表
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("vocab size:", vocab_size)

vocab_table = {token:integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab_table.items()):
    print(item)
    if i > 50:
        break

tokenizer = SimpleTokenizerV1(vocab_table)
text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab_table = {token:integer for integer, token in enumerate(all_tokens)}

print(len(vocab_table.items()))

for i, item in enumerate(list(vocab_table.items())[-5:]):
    print(item)

tokenizer = SimpleTokenizerV2(vocab_table)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)
print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))