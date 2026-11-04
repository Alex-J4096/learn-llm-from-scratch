import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

# 归一化
attn_weight_2_tmp = attn_scores_2/attn_scores_2.sum()
print("Attention weight:", attn_weight_2_tmp)
print("Sum:", attn_weight_2_tmp.sum())

# 或者用softmax来进行归一化
# def softmax_naive(x):
#     return torch.exp(x)/torch.exp(x).sum(dim=0)

attn_weight_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weight:", attn_weight_2)
print("Sum:", attn_weight_2.sum())

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weight_2[i]*x_i
print(context_vec_2)


attn_scores = inputs @ inputs.T
print(attn_scores)

attn_weight = torch.softmax(attn_scores, dim=-1)
print("Attention weight:", attn_weight)

all_context_vecs = attn_weight @ inputs
print(all_context_vecs)

