import torch
import tiktoken
import torch.nn as nn
from importlib.metadata import version
from supplementary import TransformerBlock, LayerNorm

print("torch version: ", version("torch"))
print("tiktoken version: ", version("tiktoken"))

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
}


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embs = self.token_emb(in_idx)
        pos_embs = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = token_embs + pos_embs
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


tokenizer = tiktoken.get_encoding("gpt2")
batch = []
text1 = "Every effort moves you"
text2 = "Every day hold a"

batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))

batch = torch.stack(batch, dim=0)
print("Batch: ", batch)

torch.manual_seed(101)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("DEBUG: Input: ", batch)
print("DEBUG: Output: ", out.shape)
print("DEBUG: Out: ", out)


# Generating Text
def generate_text_simple(model, idx, max_new_token, context_size):
    for _ in range(max_new_token):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


text1 = "Every effort moves you"
idx = torch.tensor(tokenizer.encode(text1)).unsqueeze(0)

response = generate_text_simple(model=model, idx=idx, max_new_token=10, context_size=1024)
print("DEBUG: response: ", response[0])
print("DEBUG: Decoded: ", tokenizer.decode(list(response[0])))
