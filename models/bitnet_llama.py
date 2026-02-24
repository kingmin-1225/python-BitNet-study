import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import math
import torch
import torch.nn.functional as F
from torch import nn
from utils import BitLinear

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(tensor: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(tensor.float().reshape(*tensor.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_complex)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*tensor.shape)
    return x_out.type_as(tensor)

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, max_seq_len=256):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.query = BitLinear(n_embd, n_embd, bias=False)
        self.key = BitLinear(n_embd, n_embd, bias=False)
        self.value = BitLinear(n_embd, n_embd, bias=False)
        self.proj = BitLinear(n_embd, n_embd, bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len), persistent=False)
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer('freqs_cis', freqs_cis)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_head, self.head_dim)
        k = self.key(x).view(B, T, self.n_head, self.head_dim)
        v = self.value(x).view(B, T, self.n_head, self.head_dim)

        q = apply_rotary_emb(q, self.freqs_cis[:T])
        k = apply_rotary_emb(k, self.freqs_cis[:T])

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.w1 = BitLinear(n_embd, 4 * n_embd, bias=False)
        self.w2 = BitLinear(n_embd, 4 * n_embd, bias=False)
        self.w3 = BitLinear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x)) 

class BitDecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, max_seq_len=256):
        super().__init__()
        self.attention = SelfAttention(n_embd, n_head, max_seq_len)
        self.feed_forward = FeedForward(n_embd)
        self.attention_norm = RMSNorm(n_embd)
        self.ffn_norm = RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class BitLLaMA(nn.Module):
    def __init__(self, vocab_size, n_embd=512, n_head=8, n_layer=8, max_seq_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.max_seq_len = max_seq_len

        self.tok_embeddings = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([BitDecoderBlock(n_embd, n_head, max_seq_len) for _ in range(n_layer)])
        self.norm = RMSNorm(n_embd)
        self.output = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx