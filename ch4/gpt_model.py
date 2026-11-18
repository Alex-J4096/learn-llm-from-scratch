import torch
from torch import nn
import tiktoken

from ch4.layer_norm import LayerNorm
from ch4.transformer import TransformerBlock
from ch4.gpt_cfg import GPT_CONFIG_MEDIUM


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])],
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# idx是当前文本的索引数组, (batch, n_tokens)
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # 截断，如果输入文本长度为10, llm仅支持长度为5的文本, 则只有最后5个token会被输入
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)

    return idx

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)

    model_med = GPTModel(GPT_CONFIG_MEDIUM)
    total_params = sum(p.numel() for p in model_med.parameters() if p.requires_grad)
    print("Total parameters:", total_params)

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    # 添加batch维度, 形状变为(1,4)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model_med.eval()
    out = generate_text_simple(
        model=model_med,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_MEDIUM["context_length"],
    )
    print("output", out)
    print("output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print("decoded_text:", decoded_text)
