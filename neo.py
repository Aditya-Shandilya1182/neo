import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False, dtype=config.d_type)
        self.w2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False, dtype=config.d_type)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x_w1 = self.w1(x)
        x = F.silu(x_w1)
        x = self.w2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding size must be divisible by the number of heads"
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False, dtype=config.d_type)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False, dtype=config.d_type)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size,
                       config.block_size, dtype=torch.bool))
        )
        self.att_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.att_dropout(y)
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = MLP(config)
        self.attention_norm = nn.RMSNorm(config.n_embd, dtype=config.d_type)
        self.ffn_norm = nn.RMSNorm(config.n_embd, dtype=config.d_type)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Neo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd, dtype=config.d_type)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd, dtype=config.d_type)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.RMSNorm(config.n_embd, dtype=config.d_type)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, dtype=config.d_type)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        tok_emb = self.token_embedding(index)
        pos_emb = self.position_embedding(
            torch.arange(T, device=self.config.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            b_s = self.config.block_size
            index_cond = index[:, -b_s:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

class ModelConfig:
    batch_size: int = 8  
    block_size: int = 512  
    max_iters: int = 10000  
    learning_rate: float = 1e-4  
    eval_iters: int = 500  
    n_embd: int = 512  
    n_head: int = 8  
    n_layer: int = 12  
    head_size: int = 512
    d_type: torch.dtype = torch.float32
    vocab_size: int = 50257
    dropout: float = 0.1  
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
