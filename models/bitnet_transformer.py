import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import math
import torch
import torch.nn.functional as F
from torch import nn
from utils import BitLinear

class BitSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.query = BitLinear(n_embd, n_embd, bias=False)
        self.key = BitLinear(n_embd, n_embd, bias=False)
        self.value = BitLinear(n_embd, n_embd, bias=False)
        self.proj = BitLinear(n_embd, n_embd)
        self.n_head = n_head
        self.register_buffer("bias", torch.tril(torch.ones(256, 256)).view(1, 1, 256, 256), persistent=False) 
        ## register_buffer를 통해 선언하는 이유
        ## self.bias와 같이 일반변수로 선언하게 되면 입력 데이터와 함께 gpu로 이동하지 못하게 됨. -> 디바이스 불일치 에러 발생
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))

        ## 위의 register_buffer를 통해 선언한 bias의 0인 부분을 -inf로 대체
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) 
        attn = F.softmax(attn, dim=-1)
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class BitBlock(nn.Module): ## Transformer block
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = BitSelfAttention(n_embd, n_head)
        self.ffwd = nn.Sequential(
            BitLinear(n_embd, 4 * n_embd),
            nn.GELU(),
            BitLinear(4 * n_embd, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BitNetLM(nn.Module):
    def __init__(self, vocab_size, n_embd=512, n_head=8, n_layer=8):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(256, n_embd) # max_len=256
        self.blocks = nn.Sequential(*[BitBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # 마지막 출력은 일반 Linear 권장

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -256:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx