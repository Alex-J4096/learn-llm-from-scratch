import torch
from torch import nn
# from causal_self_attention_class import CausalAttention
# class MultiHeadAttentionWrapper(nn.Module):
#     def __init__(self, input_dim, output_dim, context_len, dropout, num_heads, qkv_bias=False):
#         super().__init__()
#         self.heads = nn.ModuleList(
#             [CausalAttention(input_dim, output_dim, context_len, dropout, qkv_bias)
#              for _ in range(num_heads)
#             ]
#         )
#     def forward(self, x):
#         return torch.cat([head(x) for head in self.heads], dim=-1)


# inputs = torch.tensor(
#     [[0.43, 0.15, 0.89], # Your     (x^1)
#      [0.55, 0.87, 0.66], # journey  (x^2)
#      [0.57, 0.85, 0.64], # starts   (x^3)
#      [0.22, 0.58, 0.33], # with     (x^4)
#      [0.77, 0.25, 0.10], # one      (x^5)
#      [0.05, 0.80, 0.55]] # step     (x^6)
# )
# batch = torch.stack((inputs, inputs), dim=0)

# torch.manual_seed(123)
# context_len = batch.shape[1] # token的长度
# d_in, d_out = 3,1
# mha = MultiHeadAttentionWrapper(
#     d_in, d_out, context_len, 0.0, num_heads=2
# )
# context_vecs = mha(batch)
#
# print(context_vecs)
# print(context_vecs.shape)

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, context_len, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 确保Multi-Head Attention能够将输出特征维度均匀分配给每个Attention Head
        assert (output_dim % num_heads) == 0, \
        "output_dim must be divisible by num_heads"

        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.W_query = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        # 用一个线性层来组合各个头的输出
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, x):
        batch_size, num_tokens, input_dim = x.shape
        # tensor形状 (b, num_tokens, input_dim)
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        # 通过添加一个head_dim来把key向量的维度拆给多个head
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # 交换num_tokens和num_heads的维度
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # tensor形状 (b, num_tokens, n_heads, head_dim), 因为最后拼接的是多个head的context_vec
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并最后两个维度，把多头的context_vec拼接
        context_vec = context_vec.contiguous().view(batch_size, num_tokens, self.output_dim)
        # 添加一个输出层的线性投影允许模型跨 Head 混合信息
        context_vec = self.out_proj(context_vec)
        return context_vec

# torch.manual_seed(123)
# batch_size, context_len, intput_dim = batch.shape
# d_out = 2
# mha = MultiHeadAttention(intput_dim, d_out, context_len, 0.0, num_heads=2)
# context_vecs = mha(batch)
# print(context_vecs)
# print(context_vecs.shape)