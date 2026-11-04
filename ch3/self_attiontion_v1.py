import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(output_dim, input_dim))
        self.W_key = nn.Parameter(torch.rand(output_dim, input_dim))
        self.W_value = nn.Parameter(torch.rand(output_dim, input_dim))

    def forward(self, x):
        keys = x @ self.W_query
        queries = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec