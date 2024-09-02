import torch
import tiktoken
import torch.nn as nn
from supplementary import (
    TransformerBlock,
    LayerNorm,
    create_dataloader_v1,
    calc_loss_loader,
    calc_loss_batch,
    evaluate_model,
    generate_and_print_sample,
)

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
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


torch.manual_seed(101)
model = GPTModel(GPT_CONFIG_124M)
model.eval()


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(list(token_ids[0]))


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")


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


token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_token=10,
    context_size=GPT_CONFIG_124M["context_length"],
)
print("DEBUG: token_ids: ", token_ids_to_text(token_ids, tokenizer))

# Dataloaders:
with open("got-song-of-ice-and-fire.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total Text: ", len(raw_text))
print("Initial few sentences: ", raw_text[:99])

train_ratio = 0.9
split_idx = int(train_ratio * len(raw_text))
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]

torch.manual_seed(101)
train_loader = create_dataloader_v1(
    text=train_data, batch_size=2, max_length=256, stride=256, drop_last=True, shuffle=True, num_workers=0
)

val_loader = create_dataloader_v1(
    text=val_data, batch_size=2, max_length=256, stride=256, drop_last=False, shuffle=False, num_workers=0
)

print("DEBUG: Train Loader Length: ", len(train_loader))
# print("Train Loader: ")
# for x, y in train_loader:
#     print(x.shape, y.shape)
train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()
print("DEBUG: train_tokens: ", train_tokens)

print("DEBUG: Val Loader Length: ", len(val_loader))
# print("Val Loader: ")
# for x, y in val_loader:
#     print(x.shape, y.shape)
val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()
print("DEBUG: val_tokens: ", val_tokens)


# Calculate Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# with torch.no_grad():
#     train_loss = calc_loss_loader(train_loader, model, device)
#     val_loss = calc_loss_loader(val_loader, model, device)

# print("DEBUG: Training Loss: ", train_loss)
# print("DEBUG: Val Loss: ", val_loss)


def train_model_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer
):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): " f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Print a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer,
)

torch.save(model.state_dict(), "model.pth")
