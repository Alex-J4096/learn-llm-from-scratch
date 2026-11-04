import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(input_dim, output_dim))
        self.W_key = nn.Parameter(torch.rand(input_dim, output_dim))
        self.W_value = nn.Parameter(torch.rand(input_dim, output_dim))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
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

# 输入嵌入维度3，输出嵌入维度2
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))