from importlib.metadata import version
import tiktoken

print(version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

strings = tokenizer.decode(integers)
print(strings)

unknown_text = ("Akwirw ier")
token_ids = tokenizer.encode(unknown_text)
print(token_ids)

unknown_string = tokenizer.decode(token_ids)
print(unknown_string)