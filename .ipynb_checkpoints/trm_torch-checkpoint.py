"""
tiny recursive model (trm) - pytorch implementation
based on: "less is more: recursive reasoning with tiny networks"

key architecture:
- single tiny 2-layer network
- latent recursion: z = net(x, y, z) for n iterations
- answer update: y = net(y, z)
- deep supervision: carry (y, z) across supervision steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import time

# ============================================================================
# data preprocessing
# ============================================================================




class CharDataset(Dataset):
    """character-level dataset for language modeling"""
    
    def __init__(self, text, seq_len, char_to_idx=None):
        self.seq_len = seq_len
        
        # build vocab if not provided
        if char_to_idx is None:
            chars = sorted(set(text))
            self.char_to_idx = {c: i for i, c in enumerate(chars)}
            self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        else:
            self.char_to_idx = char_to_idx
            self.idx_to_char = {i: c for c, i in char_to_idx.items()}
        
        self.vocab_size = len(self.char_to_idx)
        
        # encode text
        self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


# ============================================================================
# model components
# ============================================================================

class RMSNorm(nn.Module):
    """root mean square layer normalization"""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """swiglu feedforward: swish(x @ w_gate) * (x @ w_up) @ w_down"""
    
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)
    
    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # silu = swish
        up = self.w_up(x)
        return self.w_down(gate * up)


class MLPMixer(nn.Module):
    """mlp applied on sequence dimension (token mixing)"""
    
    def __init__(self, seq_len, hidden_mult=1):
        super().__init__()
        hidden = seq_len * hidden_mult
        self.fc1 = nn.Linear(seq_len, hidden)
        self.fc2 = nn.Linear(hidden, seq_len)
    
    def forward(self, x):
        # x: [batch, seq, dim]
        x = x.transpose(1, 2)  # [batch, dim, seq]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.transpose(1, 2)  # [batch, seq, dim]


class TRMBlock(nn.Module):
    """
    single trm block: norm -> mixer -> norm -> swiglu ffn
    with residual connections
    """
    
    def __init__(self, dim, hidden_dim, seq_len, use_attention=False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        if use_attention:
            # simple single-head attention
            self.mixer = nn.MultiheadAttention(dim, num_heads=4, batch_first=True, bias=False)
            self.use_attention = True
        else:
            self.mixer = MLPMixer(seq_len)
            self.use_attention = False
        
        self.ffn = SwiGLU(dim, hidden_dim)
    
    def forward(self, x):
        # mixer block
        h = self.norm1(x)
        if self.use_attention:
            h, _ = self.mixer(h, h, h, need_weights=False)
        else:
            h = self.mixer(h)
        x = x + h
        
        # ffn block
        x = x + self.ffn(self.norm2(x))
        return x


class TRMNetwork(nn.Module):
    """
    tiny recursive network - single network for both:
    - latent update: z = net(concat(x, y, z))
    - answer update: y = net(concat(y, z))
    
    task is differentiated by presence/absence of x
    """
    
    def __init__(self, vocab_size, dim, hidden_dim, seq_len, num_layers=2, use_attention=False):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        
        # embeddings
        self.token_embed = nn.Embedding(vocab_size, dim)
        
        # learnable initial y and z
        self.y_init = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.z_init = nn.Parameter(torch.zeros(1, seq_len, dim))
        
        # input projections
        # latent: concat(x, y, z) -> 3*dim -> dim
        # answer: concat(y, z) -> 2*dim -> dim
        self.proj_latent = nn.Linear(3 * dim, dim, bias=False)
        self.proj_answer = nn.Linear(2 * dim, dim, bias=False)
        
        # transformer blocks
        self.blocks = nn.ModuleList([
            TRMBlock(dim, hidden_dim, seq_len, use_attention)
            for _ in range(num_layers)
        ])
        
        # output head
        self.output_norm = RMSNorm(dim)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        
        # q head for halting probability
        self.q_head = nn.Linear(dim, 1, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward_latent(self, x, y, z):
        """z_new = net(concat(x, y, z))"""
        h = torch.cat([x, y, z], dim=-1)
        h = self.proj_latent(h)
        for block in self.blocks:
            h = block(h)
        return h
    
    def forward_answer(self, y, z):
        """y_new = net(concat(y, z))"""
        h = torch.cat([y, z], dim=-1)
        h = self.proj_answer(h)
        for block in self.blocks:
            h = block(h)
        return h
    
    def get_logits(self, y):
        """convert embedded y to vocab logits"""
        return self.output_head(self.output_norm(y))
    
    def get_halt_prob(self, y):
        """halting probability from pooled y"""
        y_pooled = y.mean(dim=1)
        return torch.sigmoid(self.q_head(y_pooled).squeeze(-1))


class TRM(nn.Module):
    """
    tiny recursive model - full model with recursive reasoning
    
    algorithm (from paper pseudocode):
    1. latent_recursion: for n steps, z = net(x, y, z); then y = net(y, z)
    2. deep_recursion: T-1 times without grad, 1 time with grad
    3. deep_supervision: repeat for N_sup steps, detaching y,z between steps
    """
    
    def __init__(self, vocab_size, dim=128, hidden_dim=256, seq_len=32,
                 n_latent=6, t_recurse=3, n_sup=4, num_layers=2, use_attention=False):
        super().__init__()
        self.n_latent = n_latent
        self.t_recurse = t_recurse
        self.n_sup = n_sup
        
        self.net = TRMNetwork(
            vocab_size, dim, hidden_dim, seq_len,
            num_layers=num_layers, use_attention=use_attention
        )
    
    def latent_recursion(self, x, y, z):
        """single recursion block: n latent updates, then 1 answer update"""
        for _ in range(self.n_latent):
            z = self.net.forward_latent(x, y, z)
        y = self.net.forward_answer(y, z)
        return y, z
    
    def deep_recursion(self, x, y, z):
        """
        T recursions total:
        - T-1 without gradients (use torch.no_grad)
        - 1 with gradients (backprop through this)
        """
        # T-1 recursions without gradient
        with torch.no_grad():
            for _ in range(self.t_recurse - 1):
                y, z = self.latent_recursion(x, y, z)
        
        # final recursion with gradient
        y, z = self.latent_recursion(x, y, z)
        
        return y.detach(), z.detach(), y  # return detached for next step, non-detached for loss
    
    def forward(self, x_idx, return_all_steps=False):
        """
        full forward with deep supervision
        
        x_idx: [batch, seq] token indices
        returns: logits [batch, seq, vocab] from final supervision step
        """
        batch = x_idx.shape[0]
        device = x_idx.device
        
        # embed input
        x = self.net.token_embed(x_idx)
        
        # initialize y, z
        y = self.net.y_init.expand(batch, -1, -1)
        z = self.net.z_init.expand(batch, -1, -1)
        
        all_logits = []
        
        # deep supervision loop
        for step in range(self.n_sup):
            y_detached, z_detached, y_for_loss = self.deep_recursion(x, y, z)
            
            logits = self.net.get_logits(y_for_loss)
            all_logits.append(logits)
            
            # use detached for next step
            y, z = y_detached, z_detached
        
        if return_all_steps:
            return all_logits
        return all_logits[-1]  # return final step logits
    
    @torch.no_grad()
    def generate(self, seed_idx, max_len=100, temperature=1.0):
        """
        generate tokens autoregressively
        
        seed_idx: [seq_len] starting token indices
        """
        self.eval()
        device = next(self.parameters()).device
        
        tokens = seed_idx.tolist() if isinstance(seed_idx, torch.Tensor) else list(seed_idx)
        seq_len = self.net.seq_len
        
        for _ in range(max_len):
            # prepare input
            if len(tokens) < seq_len:
                x = [0] * (seq_len - len(tokens)) + tokens
            else:
                x = tokens[-seq_len:]
            
            x = torch.tensor([x], dtype=torch.long, device=device)
            
            # forward
            logits = self.forward(x)  # [1, seq, vocab]
            
            # sample from last position
            probs = F.softmax(logits[0, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            tokens.append(next_token)
        
        return tokens


# ============================================================================
# training
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, accumulation_steps=1, print_every=10):
    """train for one epoch with optional gradient accumulation"""
    model.train()
    total_loss = 0
    total_tokens = 0

    optimizer.zero_grad()

    num_batches = len(dataloader)

    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # forward - get all supervision step logits
        all_logits = model(x, return_all_steps=True)

        # compute loss over all supervision steps (deep supervision)
        loss = 0
        for logits in all_logits:
            step_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss + step_loss
        loss = loss / len(all_logits)

        # backward with accumulation
        loss_scaled = loss / accumulation_steps
        loss_scaled.backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # track actual loss (not scaled)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

        # progress logging
        if (step + 1) % print_every == 0:
            avg_loss = total_loss / total_tokens
            print(f"  batch {step + 1:3d}/{num_batches} | loss: {loss.item():.4f} | avg: {avg_loss:.4f}")

    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, dataloader, device):
    """evaluate model, return loss and accuracy"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
        
        preds = logits.argmax(dim=-1)
        correct = (preds == y).sum().item()
        
        total_loss += loss.item()
        total_correct += correct
        total_tokens += y.numel()
    
    return total_loss / total_tokens, total_correct / total_tokens


def count_parameters(model):
    """count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# main
# ============================================================================

def main():
    # hyperparameters
    seq_len = 32
    dim = 64
    hidden_dim = 128
    n_latent = 3
    t_recurse = 2
    n_sup = 2
    num_layers = 2
    use_attention = False  # mlp mixer (paper shows it works better for small fixed context)

    batch_size = 32
    epochs = 5
    lr = 1e-3

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")

    print("=" * 60)
    print("tiny recursive model (trm) - pytorch implementation")
    print("=" * 60)

    # load data from shakespeare.txt
    filepath = "shakespeare.txt"
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"loaded {len(text)} characters")

    # create dataset
    dataset = CharDataset(text, seq_len)
    vocab_size = dataset.vocab_size
    print(f"vocab size: {vocab_size}")
    
    # train/test split
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"train: {train_size}, test: {test_size}")

    # create model
    model = TRM(
        vocab_size=vocab_size,
        dim=dim,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        n_latent=n_latent,
        t_recurse=t_recurse,
        n_sup=n_sup,
        num_layers=num_layers,
        use_attention=use_attention
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"parameters: {num_params:,}")
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # training loop
    print("\ntraining...")
    print(f"samples: {train_size}, batch_size: {batch_size}, steps/epoch: {len(train_loader)}")
    best_loss = float('inf')

    for epoch in range(epochs):
        start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device, print_every=5)
        test_loss, test_acc = evaluate(model, test_loader, device)

        scheduler.step()

        elapsed = time.time() - start

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'trm_best.pt')

        print(f"epoch {epoch+1:2d}/{epochs} complete | "
              f"train_loss: {train_loss:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_acc: {test_acc:.4f} | "
              f"time: {elapsed:.1f}s")
    
    # load best model
    model.load_state_dict(torch.load('trm_best.pt'))
    
    # final evaluation
    print("\nfinal evaluation...")
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"best test loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")
    
    # generation
    print("\ngenerating text...")
    seed = "ROMEO:"
    seed_idx = [dataset.char_to_idx.get(c, 0) for c in seed]
    # pad to seq_len
    seed_idx = [0] * (seq_len - len(seed_idx)) + seed_idx
    seed_tensor = torch.tensor(seed_idx, dtype=torch.long, device=device)
    
    generated_idx = model.generate(seed_tensor, max_len=200, temperature=0.8)
    generated_text = ''.join([dataset.idx_to_char.get(i, '?') for i in generated_idx])
    
    print(f"seed: '{seed}'")
    print(f"generated:\n{generated_text[-200:]}")  # show last 200 chars
    
    print("\ndone!")


if __name__ == "__main__":
    main()