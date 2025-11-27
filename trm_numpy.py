"""
tiny recursive model (trm) implementation in numpy
based on: "less is more: recursive reasoning with tiny networks"

key architecture:
- single tiny 2-layer network
- recursive latent updates: z = net(x, y, z) for n iterations
- answer refinement: y = net(y, z)
- deep supervision: carry (y, z) across supervision steps
"""

import numpy as np
import urllib.request
import os
import pickle
from collections import Counter

# ============================================================================
# data preprocessing - tiny shakespeare (character-level)
# ============================================================================

def download_tiny_shakespeare(filepath="tiny_shakespeare.txt"):
    """download tiny shakespeare dataset if not present"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not os.path.exists(filepath):
        print(f"downloading tiny shakespeare to {filepath}...")
        urllib.request.urlretrieve(url, filepath)
    return filepath


def load_and_preprocess(filepath, seq_len=32, vocab_size=None):
    """
    load text, build vocab, create sequences
    
    returns:
        x_train: input sequences [num_seqs, seq_len]
        y_train: target sequences (shifted by 1) [num_seqs, seq_len]
        char_to_idx: character to index mapping
        idx_to_char: index to character mapping
    """
    # load raw text
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"loaded {len(text)} characters")
    
    # build vocabulary from character frequencies
    char_freq = Counter(text)
    
    # if vocab_size specified, keep most common chars
    if vocab_size is not None:
        most_common = char_freq.most_common(vocab_size - 1)
        chars = [c for c, _ in most_common]
    else:
        chars = sorted(char_freq.keys())
    
    # add unknown token
    unk_token = '\x00'  # null char as unknown
    if unk_token not in chars:
        chars.append(unk_token)
    
    # create mappings
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    
    actual_vocab_size = len(chars)
    print(f"vocabulary size: {actual_vocab_size}")
    
    # encode text to indices
    # replace unknown chars with unk token
    unk_idx = char_to_idx[unk_token]
    encoded = np.array([char_to_idx.get(c, unk_idx) for c in text], dtype=np.int32)
    
    # create sequences
    num_seqs = (len(encoded) - 1) // seq_len
    x_data = encoded[:num_seqs * seq_len].reshape(num_seqs, seq_len)
    y_data = encoded[1:num_seqs * seq_len + 1].reshape(num_seqs, seq_len)
    
    print(f"created {num_seqs} sequences of length {seq_len}")
    
    return x_data, y_data, char_to_idx, idx_to_char


def create_batches(x_data, y_data, batch_size, shuffle=True):
    """yield batches from data"""
    num_samples = len(x_data)
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_idx = indices[start:end]
        yield x_data[batch_idx], y_data[batch_idx]


# ============================================================================
# activation functions and utilities
# ============================================================================

def softmax(x, axis=-1):
    """numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sigmoid(x):
    """numerically stable sigmoid"""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def swiglu(x, w1, w2, b1=None, b2=None):
    """
    swiglu activation: swish(x @ w1) * (x @ w2)
    swish(x) = x * sigmoid(x)
    
    x: [batch, seq, dim]
    w1, w2: [dim, hidden_dim]
    """
    gate = x @ w1
    if b1 is not None:
        gate = gate + b1
    
    up = x @ w2
    if b2 is not None:
        up = up + b2
    
    # swish activation on gate
    swish_gate = gate * sigmoid(gate)
    
    return swish_gate * up


def rms_norm(x, weight, eps=1e-6):
    """
    root mean square layer normalization
    x: [..., dim]
    weight: [dim]
    """
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def cross_entropy_loss(logits, targets):
    """
    cross entropy loss for sequences
    logits: [batch, seq, vocab]
    targets: [batch, seq] integer indices
    """
    batch, seq_len, vocab = logits.shape
    
    # flatten
    logits_flat = logits.reshape(-1, vocab)
    targets_flat = targets.reshape(-1)
    
    # compute softmax
    probs = softmax(logits_flat, axis=-1)
    
    # gather probabilities for correct classes
    # clip for numerical stability
    correct_probs = probs[np.arange(len(targets_flat)), targets_flat]
    correct_probs = np.clip(correct_probs, 1e-10, 1.0)
    
    # negative log likelihood
    loss = -np.mean(np.log(correct_probs))
    
    return loss, probs.reshape(batch, seq_len, vocab)


# ============================================================================
# trm network components (2-layer transformer-like block)
# ============================================================================

class TRMLayer:
    """
    single trm layer: rms_norm -> attention/mlp -> rms_norm -> swiglu ffn
    
    simplified: using mlp mixing instead of attention (as paper suggests
    mlp works better for small fixed context)
    """
    
    def __init__(self, dim, hidden_dim, seq_len):
        """
        dim: embedding dimension
        hidden_dim: feedforward hidden dimension
        seq_len: sequence length (for mlp mixer)
        """
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # scale for xavier init
        scale_dim = np.sqrt(2.0 / dim)
        scale_hidden = np.sqrt(2.0 / hidden_dim)
        scale_seq = np.sqrt(2.0 / seq_len)
        
        # mlp mixer weights (applied on sequence dimension)
        # projects [batch, seq, dim] -> transpose -> mlp on seq -> transpose back
        self.w_mix1 = np.random.randn(seq_len, seq_len) * scale_seq
        self.w_mix2 = np.random.randn(seq_len, seq_len) * scale_seq
        
        # rms norm weights
        self.norm1_weight = np.ones(dim)
        self.norm2_weight = np.ones(dim)
        
        # swiglu ffn weights
        self.w_gate = np.random.randn(dim, hidden_dim) * scale_dim
        self.w_up = np.random.randn(dim, hidden_dim) * scale_dim
        self.w_down = np.random.randn(hidden_dim, dim) * scale_hidden
        
        # gradients storage
        self.grads = {}
        
        # cache for backward pass
        self.cache = {}
    
    def forward(self, x):
        """
        x: [batch, seq, dim]
        returns: [batch, seq, dim]
        """
        batch, seq, dim = x.shape
        
        # mlp mixer block
        # transpose to [batch, dim, seq], apply mlp, transpose back
        x_norm1 = rms_norm(x, self.norm1_weight)
        self.cache['x_norm1'] = x_norm1
        self.cache['x_input'] = x
        
        # mixer: [batch, seq, dim] -> [batch, dim, seq]
        x_t = np.transpose(x_norm1, (0, 2, 1))
        
        # two-layer mlp on sequence dimension
        h_mix = x_t @ self.w_mix1  # [batch, dim, seq]
        h_mix = np.maximum(h_mix, 0)  # relu
        self.cache['h_mix_pre_relu'] = x_t @ self.w_mix1
        h_mix_out = h_mix @ self.w_mix2
        
        # transpose back and residual
        mix_out = np.transpose(h_mix_out, (0, 2, 1))
        x = x + mix_out
        self.cache['x_after_mix'] = x
        self.cache['mix_out'] = mix_out
        
        # ffn block
        x_norm2 = rms_norm(x, self.norm2_weight)
        self.cache['x_norm2'] = x_norm2
        
        # swiglu
        gate = x_norm2 @ self.w_gate
        up = x_norm2 @ self.w_up
        swish_gate = gate * sigmoid(gate)
        ffn_hidden = swish_gate * up
        self.cache['gate'] = gate
        self.cache['up'] = up
        self.cache['swish_gate'] = swish_gate
        self.cache['ffn_hidden'] = ffn_hidden
        
        ffn_out = ffn_hidden @ self.w_down
        
        # residual
        out = x + ffn_out
        self.cache['ffn_out'] = ffn_out
        
        return out
    
    def backward(self, grad_out):
        """
        backward pass through layer
        grad_out: gradient from upstream [batch, seq, dim]
        returns: gradient w.r.t input x
        """
        # ffn residual: grad flows to both ffn_out and x_after_mix
        grad_ffn_out = grad_out
        grad_x_after_mix = grad_out.copy()
        
        # w_down backward
        ffn_hidden = self.cache['ffn_hidden']
        self.grads['w_down'] = np.tensordot(ffn_hidden, grad_ffn_out, axes=([0,1], [0,1]))
        grad_ffn_hidden = grad_ffn_out @ self.w_down.T
        
        # swiglu backward
        gate = self.cache['gate']
        up = self.cache['up']
        swish_gate = self.cache['swish_gate']
        
        # d(swish_gate * up) / d(swish_gate) = up
        # d(swish_gate * up) / d(up) = swish_gate
        grad_swish_gate = grad_ffn_hidden * up
        grad_up = grad_ffn_hidden * swish_gate
        
        # swish backward: d(x * sigmoid(x)) / dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        sig_gate = sigmoid(gate)
        grad_gate = grad_swish_gate * (sig_gate + gate * sig_gate * (1 - sig_gate))
        
        # w_gate, w_up backward
        x_norm2 = self.cache['x_norm2']
        self.grads['w_gate'] = np.tensordot(x_norm2, grad_gate, axes=([0,1], [0,1]))
        self.grads['w_up'] = np.tensordot(x_norm2, grad_up, axes=([0,1], [0,1]))
        
        grad_x_norm2 = grad_gate @ self.w_gate.T + grad_up @ self.w_up.T
        
        # rms_norm backward (simplified - ignoring weight gradient for now)
        x_after_mix = self.cache['x_after_mix']
        rms = np.sqrt(np.mean(x_after_mix ** 2, axis=-1, keepdims=True) + 1e-6)
        grad_x_from_ffn = (grad_x_norm2 * self.norm2_weight) / rms
        
        grad_x_after_mix = grad_x_after_mix + grad_x_from_ffn
        
        # mixer residual
        grad_mix_out = grad_x_after_mix
        grad_x_input = grad_x_after_mix.copy()
        
        # transpose backward
        grad_h_mix_out = np.transpose(grad_mix_out, (0, 2, 1))
        
        # w_mix2 backward
        h_mix_pre_relu = self.cache['h_mix_pre_relu']
        h_mix = np.maximum(h_mix_pre_relu, 0)
        self.grads['w_mix2'] = np.tensordot(h_mix, grad_h_mix_out, axes=([0,1], [0,1]))
        grad_h_mix = grad_h_mix_out @ self.w_mix2.T
        
        # relu backward
        grad_h_mix_pre = grad_h_mix * (h_mix_pre_relu > 0).astype(float)
        
        # w_mix1 backward
        x_norm1 = self.cache['x_norm1']
        x_t = np.transpose(x_norm1, (0, 2, 1))
        self.grads['w_mix1'] = np.tensordot(x_t, grad_h_mix_pre, axes=([0,1], [0,1]))
        grad_x_t = grad_h_mix_pre @ self.w_mix1.T
        
        grad_x_norm1 = np.transpose(grad_x_t, (0, 2, 1))
        
        # rms_norm backward for norm1
        x_input = self.cache['x_input']
        rms1 = np.sqrt(np.mean(x_input ** 2, axis=-1, keepdims=True) + 1e-6)
        grad_x_from_mix = (grad_x_norm1 * self.norm1_weight) / rms1
        
        grad_x_input = grad_x_input + grad_x_from_mix
        
        return grad_x_input
    
    def get_params(self):
        """return list of (param, grad) tuples"""
        return [
            (self.w_mix1, self.grads.get('w_mix1', np.zeros_like(self.w_mix1))),
            (self.w_mix2, self.grads.get('w_mix2', np.zeros_like(self.w_mix2))),
            (self.w_gate, self.grads.get('w_gate', np.zeros_like(self.w_gate))),
            (self.w_up, self.grads.get('w_up', np.zeros_like(self.w_up))),
            (self.w_down, self.grads.get('w_down', np.zeros_like(self.w_down))),
        ]


class TRMNetwork:
    """
    tiny recursive model network
    
    single network used for both:
    - latent update: z = net(concat(x, y, z))
    - answer update: y = net(concat(y, z))
    
    the task is differentiated by presence/absence of x in input
    """
    
    def __init__(self, vocab_size, dim, hidden_dim, seq_len, num_layers=2):
        """
        vocab_size: number of tokens in vocabulary
        dim: embedding dimension
        hidden_dim: ffn hidden dimension
        seq_len: sequence length
        num_layers: number of transformer layers (paper uses 2)
        """
        self.vocab_size = vocab_size
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        
        # embeddings
        scale = np.sqrt(2.0 / dim)
        self.token_embed = np.random.randn(vocab_size, dim) * 0.02
        
        # learned initial embeddings for y and z
        self.y_init = np.zeros((1, seq_len, dim))
        self.z_init = np.zeros((1, seq_len, dim))
        
        # projection layers for concatenated input
        # for latent update: concat(x, y, z) -> dim, so 3*dim -> dim
        # for answer update: concat(y, z) -> dim, so 2*dim -> dim
        self.proj_latent = np.random.randn(3 * dim, dim) * np.sqrt(2.0 / (3 * dim))
        self.proj_answer = np.random.randn(2 * dim, dim) * np.sqrt(2.0 / (2 * dim))
        
        # transformer layers
        self.layers = [TRMLayer(dim, hidden_dim, seq_len) for _ in range(num_layers)]
        
        # output head
        self.output_head = np.random.randn(dim, vocab_size) * scale
        
        # q head for halting (single scalar output)
        self.q_head = np.random.randn(dim, 1) * scale
        
        # gradient storage
        self.grads = {}
        self.cache = {}
    
    def embed(self, x_idx):
        """
        embed token indices
        x_idx: [batch, seq] integer indices
        returns: [batch, seq, dim]
        """
        return self.token_embed[x_idx]
    
    def forward_latent(self, x, y, z):
        """
        latent update: z_new = net(concat(x, y, z))
        
        x, y, z: [batch, seq, dim]
        returns: z_new [batch, seq, dim]
        """
        # concatenate along feature dimension
        concat_in = np.concatenate([x, y, z], axis=-1)  # [batch, seq, 3*dim]
        
        # project to dim
        h = concat_in @ self.proj_latent  # [batch, seq, dim]
        self.cache['latent_concat'] = concat_in
        self.cache['latent_h0'] = h.copy()
        
        # pass through layers
        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            self.cache[f'latent_h{i+1}'] = h.copy()
        
        return h  # new z
    
    def forward_answer(self, y, z):
        """
        answer update: y_new = net(concat(y, z))
        
        y, z: [batch, seq, dim]
        returns: y_new [batch, seq, dim]
        """
        # concatenate
        concat_in = np.concatenate([y, z], axis=-1)  # [batch, seq, 2*dim]
        
        # project
        h = concat_in @ self.proj_answer  # [batch, seq, dim]
        self.cache['answer_concat'] = concat_in
        self.cache['answer_h0'] = h.copy()
        
        # pass through layers
        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            self.cache[f'answer_h{i+1}'] = h.copy()
        
        return h  # new y (embedded)
    
    def get_logits(self, y_embed):
        """
        convert embedded y to logits
        y_embed: [batch, seq, dim]
        returns: logits [batch, seq, vocab]
        """
        return y_embed @ self.output_head
    
    def get_halt_prob(self, y_embed):
        """
        get halting probability from y embedding
        y_embed: [batch, seq, dim]
        returns: halt_prob [batch]
        """
        # pool over sequence (mean)
        y_pooled = np.mean(y_embed, axis=1)  # [batch, dim]
        q = y_pooled @ self.q_head  # [batch, 1]
        return sigmoid(q.squeeze(-1))


class TRM:
    """
    tiny recursive model - full training/inference wrapper
    
    algorithm:
    1. latent recursion: for i in range(n): z = net(x, y, z)
    2. answer update: y = net(y, z)
    3. deep recursion: repeat (1,2) T times (T-1 without grad, 1 with grad)
    4. deep supervision: repeat (3) for N_sup steps, carrying (y, z)
    """
    
    def __init__(self, vocab_size, dim=128, hidden_dim=256, seq_len=32,
                 n_latent=6, t_recurse=3, n_sup=4, lr=1e-3):
        """
        vocab_size: vocabulary size
        dim: embedding dimension
        hidden_dim: ffn hidden dimension
        seq_len: sequence length
        n_latent: number of latent update steps per recursion
        t_recurse: number of full recursions (T-1 without grad, 1 with grad)
        n_sup: number of supervision steps
        lr: learning rate
        """
        self.net = TRMNetwork(vocab_size, dim, hidden_dim, seq_len)
        self.n_latent = n_latent
        self.t_recurse = t_recurse
        self.n_sup = n_sup
        self.lr = lr
        self.seq_len = seq_len
        self.dim = dim
        
        # adam optimizer state
        self.m = {}  # first moment
        self.v = {}  # second moment
        self.t = 0   # timestep
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        
        # ema for weights (paper uses 0.999)
        self.ema_decay = 0.999
        self.ema_weights = None
    
    def latent_recursion(self, x, y, z):
        """
        single latent recursion block:
        - update z n times
        - update y once
        
        returns: (y_new, z_new)
        """
        # n latent updates
        for _ in range(self.n_latent):
            z = self.net.forward_latent(x, y, z)
        
        # answer update
        y = self.net.forward_answer(y, z)
        
        return y, z
    
    def deep_recursion(self, x, y, z):
        """
        deep recursion: T-1 times without backprop, 1 time with backprop
        
        in numpy we don't have autograd, so we just do all forward passes
        and only compute gradients for the last recursion
        
        returns: (y_final, z_final)
        """
        # T-1 recursions (no gradients needed - just forward)
        for _ in range(self.t_recurse - 1):
            y, z = self.latent_recursion(x, y, z)
        
        # final recursion (this is where we'd compute gradients)
        y, z = self.latent_recursion(x, y, z)
        
        return y, z
    
    def forward(self, x_idx):
        """
        full forward pass with deep supervision
        
        x_idx: [batch, seq] token indices
        returns: logits [batch, seq, vocab]
        """
        batch = x_idx.shape[0]
        
        # embed input
        x = self.net.embed(x_idx)
        
        # initialize y and z
        y = np.tile(self.net.y_init, (batch, 1, 1))
        z = np.tile(self.net.z_init, (batch, 1, 1))
        
        # deep supervision: run multiple supervision steps
        for step in range(self.n_sup):
            y, z = self.deep_recursion(x, y, z)
            # detach y, z for next step (in numpy, we just continue)
        
        # get output logits
        logits = self.net.get_logits(y)
        
        return logits
    
    def compute_loss(self, x_idx, y_idx):
        """
        compute cross entropy loss
        
        x_idx: [batch, seq] input tokens
        y_idx: [batch, seq] target tokens
        returns: loss scalar
        """
        logits = self.forward(x_idx)
        loss, probs = cross_entropy_loss(logits, y_idx)
        return loss, logits, probs
    
    def backward_simple(self, x_idx, y_idx, logits, probs):
        """
        simplified backward pass - compute gradients for output head
        and propagate back through network
        
        this is a simplified version that updates key parameters
        """
        batch, seq_len, vocab = logits.shape
        
        # gradient of cross entropy w.r.t logits
        # d_loss/d_logits = probs - one_hot(targets)
        grad_logits = probs.copy()
        grad_logits[np.arange(batch)[:, None], np.arange(seq_len), y_idx] -= 1
        grad_logits /= (batch * seq_len)
        
        # gradient for output head
        # logits = y_embed @ output_head
        # y_embed is the final y after all recursions
        
        # we need to get y_embed from the last forward pass
        # for simplicity, recompute
        x = self.net.embed(x_idx)
        y = np.tile(self.net.y_init, (batch, 1, 1))
        z = np.tile(self.net.z_init, (batch, 1, 1))
        
        for _ in range(self.n_sup):
            y, z = self.deep_recursion(x, y, z)
        
        y_embed = y  # final embedded answer
        
        # gradient for output_head: d_loss/d_output_head = y_embed.T @ grad_logits
        # y_embed: [batch, seq, dim], grad_logits: [batch, seq, vocab]
        grad_output_head = np.tensordot(y_embed, grad_logits, axes=([0, 1], [0, 1]))
        
        # gradient flowing back to y_embed
        grad_y_embed = grad_logits @ self.net.output_head.T  # [batch, seq, dim]
        
        # propagate through the final answer update layer
        # this is complex for full backprop, so we do a simplified update
        # focusing on the projection and layer weights
        
        self.net.grads['output_head'] = grad_output_head
        
        # simplified: also update token embeddings based on input gradient
        # this helps the model learn meaningful representations
        grad_token_embed = np.zeros_like(self.net.token_embed)
        np.add.at(grad_token_embed, x_idx.flatten(), 
                  (grad_y_embed * 0.1).reshape(-1, self.dim))  # small gradient to embeddings
        self.net.grads['token_embed'] = grad_token_embed
        
        # update projection layers with approximate gradients
        # proj_answer gradient
        concat_yz = np.concatenate([y_embed, z], axis=-1)
        grad_proj_answer = np.tensordot(concat_yz, grad_y_embed, axes=([0, 1], [0, 1]))
        self.net.grads['proj_answer'] = grad_proj_answer * 0.1
        
        return grad_logits
    
    def update_params(self):
        """update parameters using adam optimizer"""
        self.t += 1
        
        params_grads = [
            ('output_head', self.net.output_head, self.net.grads.get('output_head')),
            ('token_embed', self.net.token_embed, self.net.grads.get('token_embed')),
            ('proj_answer', self.net.proj_answer, self.net.grads.get('proj_answer')),
            ('proj_latent', self.net.proj_latent, self.net.grads.get('proj_latent', np.zeros_like(self.net.proj_latent))),
        ]
        
        # add layer parameters
        for i, layer in enumerate(self.net.layers):
            for name, (param, grad) in zip(
                ['w_mix1', 'w_mix2', 'w_gate', 'w_up', 'w_down'],
                layer.get_params()
            ):
                params_grads.append((f'layer{i}_{name}', param, grad))
        
        for name, param, grad in params_grads:
            if grad is None:
                continue
            
            # initialize moment estimates if needed
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
            
            # update moments
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            # bias correction
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # update
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def train_step(self, x_batch, y_batch):
        """single training step"""
        loss, logits, probs = self.compute_loss(x_batch, y_batch)
        self.backward_simple(x_batch, y_batch, logits, probs)
        self.update_params()
        return loss
    
    def generate(self, seed_text, char_to_idx, idx_to_char, max_len=100, temperature=1.0):
        """
        generate text from seed
        
        seed_text: initial text string
        char_to_idx: character to index mapping
        idx_to_char: index to character mapping
        max_len: maximum generation length
        temperature: sampling temperature
        """
        # encode seed
        unk_idx = char_to_idx.get('\x00', 0)
        tokens = [char_to_idx.get(c, unk_idx) for c in seed_text]
        
        generated = list(seed_text)
        
        for _ in range(max_len):
            # prepare input (use last seq_len tokens)
            if len(tokens) < self.seq_len:
                # pad with zeros at beginning
                x = np.zeros((1, self.seq_len), dtype=np.int32)
                x[0, -len(tokens):] = tokens
            else:
                x = np.array([tokens[-self.seq_len:]], dtype=np.int32)
            
            # forward pass
            logits = self.forward(x)  # [1, seq, vocab]
            
            # get last position logits
            last_logits = logits[0, -1, :] / temperature
            
            # sample
            probs = softmax(last_logits)
            next_token = np.random.choice(len(probs), p=probs)
            
            tokens.append(next_token)
            generated.append(idx_to_char.get(next_token, '?'))
        
        return ''.join(generated)


# ============================================================================
# training loop
# ============================================================================

def train(model, x_train, y_train, epochs=10, batch_size=32, print_every=100):
    """
    training loop
    
    model: TRM instance
    x_train: input sequences [num_seqs, seq_len]
    y_train: target sequences [num_seqs, seq_len]
    epochs: number of training epochs
    batch_size: batch size
    print_every: print loss every n steps
    """
    num_samples = len(x_train)
    steps_per_epoch = num_samples // batch_size
    
    print(f"training for {epochs} epochs")
    print(f"samples: {num_samples}, batch_size: {batch_size}, steps/epoch: {steps_per_epoch}")
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        step = 0
        
        for x_batch, y_batch in create_batches(x_train, y_train, batch_size):
            loss = model.train_step(x_batch, y_batch)
            epoch_loss += loss
            losses.append(loss)
            step += 1
            
            if step % print_every == 0:
                avg_loss = epoch_loss / step
                print(f"epoch {epoch+1}/{epochs}, step {step}/{steps_per_epoch}, loss: {loss:.4f}, avg: {avg_loss:.4f}")
        
        avg_epoch_loss = epoch_loss / step
        print(f"epoch {epoch+1} complete, avg loss: {avg_epoch_loss:.4f}")
    
    return losses


def evaluate(model, x_test, y_test, batch_size=32):
    """
    evaluate model on test set
    
    returns: average loss, accuracy
    """
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    
    for x_batch, y_batch in create_batches(x_test, y_test, batch_size, shuffle=False):
        logits = model.forward(x_batch)
        loss, probs = cross_entropy_loss(logits, y_batch)
        
        # compute accuracy
        preds = np.argmax(logits, axis=-1)
        correct = (preds == y_batch).sum()
        
        total_loss += loss
        total_correct += correct
        total_tokens += y_batch.size
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_tokens
    
    return avg_loss, accuracy


# ============================================================================
# main
# ============================================================================

def main():
    # hyperparameters
    seq_len = 32
    vocab_size = 100  # character-level, small vocab
    dim = 64
    hidden_dim = 128
    n_latent = 3      # reduced for faster training
    t_recurse = 2     # reduced for faster training
    n_sup = 2         # reduced for faster training
    lr = 1e-3
    batch_size = 32
    epochs = 5
    
    print("=" * 60)
    print("tiny recursive model (trm) - numpy implementation")
    print("=" * 60)
    
    # download and preprocess data
    filepath = "shakespeare.txt"
    x_data, y_data, char_to_idx, idx_to_char = load_and_preprocess(
        filepath, seq_len=seq_len, vocab_size=vocab_size
    )
    
    actual_vocab_size = len(char_to_idx)
    print(char_to_idx)
    
    # train/test split (90/10)
    split_idx = int(0.9 * len(x_data))
    x_train, x_test = x_data[:split_idx], x_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]
    
    print(f"train: {len(x_train)}, test: {len(x_test)}")
    
    # create model
    print("\ncreating model...")
    model = TRM(
        vocab_size=actual_vocab_size,
        dim=dim,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        n_latent=n_latent,
        t_recurse=t_recurse,
        n_sup=n_sup,
        lr=lr
    )
    
    # count parameters
    total_params = (
        model.net.token_embed.size +
        model.net.y_init.size +
        model.net.z_init.size +
        model.net.proj_latent.size +
        model.net.proj_answer.size +
        model.net.output_head.size +
        model.net.q_head.size +
        sum(
            layer.w_mix1.size + layer.w_mix2.size +
            layer.w_gate.size + layer.w_up.size + layer.w_down.size +
            layer.norm1_weight.size + layer.norm2_weight.size
            for layer in model.net.layers
        )
    )
    print(f"total parameters: {total_params:,}")
    
    # training
    print("\ntraining...")
    losses = train(model, x_train, y_train, epochs=epochs, batch_size=batch_size, print_every=50)
    
    # evaluation
    print("\nevaluating...")
    test_loss, test_acc = evaluate(model, x_test, y_test, batch_size=batch_size)
    print(f"test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")
    
    # generation
    print("\ngenerating text...")
    seed = "ROMEO:"
    generated = model.generate(seed, char_to_idx, idx_to_char, max_len=200, temperature=0.8)
    print(f"seed: '{seed}'")
    print(f"generated:\n{generated}")
    
    print("\ndone!")


if __name__ == "__main__":
    main()