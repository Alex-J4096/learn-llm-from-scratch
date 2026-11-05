import torch
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    def __init__(self, input_dim, output_dim, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_keys = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_values = nn.Linear(input_dim, output_dim, bias=qkv_bias)


    def forward(self, x):
        keys = self.W_keys(x)
        queries = self.W_query(x)
        values= self.W_values(x)
        attn_scores= queries @ keys.T

        context_len = attn_scores.shape[0]
        mask = torch.triu(torch.ones(context_len, context_len), diagonal=1)
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        print(masked)
        attn_weights = torch.softmax(
            masked/keys.shape[-1]**0.5, dim=-1
        )
        print(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

torch.manual_seed(789)
# 输入嵌入维度3，输出嵌入维度2
d_in = inputs.shape[1]
d_out = 2

sa_v2 = SelfAttention_v2(d_in, d_out)
sa_v2.forward(inputs)

# 利用dropout掩码额外的attention weight
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6,6)
print(dropout(example))


