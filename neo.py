import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from utils import apply_rotary_emb, precompute_freqs_cis, reshape_for_broadcast

class RMSNorm(nn.Module):
  def __init__(self, dim, norm_eps):
    super().__init__()
    self.norm_eps = norm_eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)

  def forward(self, x):
    out = self._norm(x.float()).type_as(x)
    return out * self.weight
  
class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.w1 = nn.Linear(config.embd, config.hidden_dim, bias=False)
    self.w2 = nn.Linear(config.hidden_dim, config.embd, bias=False)
    self.w3 = nn.Linear(config.embd, config.hidden_dim, bias=False)

  def forward(self, x):
    x_w1 = self.w1(x)
    x_w3 = self.w3(x)
    x = F.silu(x_w1) * x_w3
    x = self.w2(x)
    return x

class Attention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.wq = nn.Linear(config.embd, config.n_heads * config.head_dim, bias=False)
    self.wk = nn.Linear(config.embd, config.n_kv_head * config.head_dim, bias=False)
    self.wv = nn.Linear(config.embd, config.n_kv_head * config.head_dim, bias=False)
    self.wo = nn.Linear(config.n_heads * config.head_dim, config.embd, bias=False)

    self.cache_k = torch.zeros((config.batch_size, config.seq_len, config.n_kv_head, config.head_dim))
    self.cache_v = torch.zeros((config.batch_size, config.seq_len, config.n_kv_head, config.head_dim))

    self.n_heads = config.n_heads
    self.head_dim = config.head_dim
    self.n_kv_head = config.n_kv_head
    self.n_kv_head_rep = config.n_kv_head_rep

  def forward(self, x, start_pos, freqs_cis, mask):
    batch_sz, seqlen, _ = x.shape

    queries = self.wq(x)
    keys = self.wk(x)
    values = self.wv(x)

    queries = queries.view(batch_sz, seqlen, self.n_heads, self.head_dim)
    keys = keys.view(batch_sz, seqlen, self.n_kv_head, self.head_dim)
    values = values.view(batch_sz, seqlen, self.n_kv_head, self.head_dim)

    queries, keys = apply_rotary_emb(queries, keys, freqs_cis)

    self.cache_k = self.cache_k.to(queries.device)
    self.cache_v = self.cache_v.to(queries.device)

    self.cache_k[:batch_sz, start_pos : start_pos + seqlen] = keys
    self.cache_v[:batch_sz, start_pos : start_pos + seqlen] = values

    keys = self.cache_k[:batch_sz, : start_pos + seqlen]
    values = self.cache_v[:batch_sz, : start_pos + seqlen]

    keys = torch.repeat_interleave(keys, dim=2, repeats=self.n_kv_head_rep)
    values = torch.repeat_interleave(values, dim=2, repeats=self.n_kv_head_rep)

    queries = queries.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    out = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask)
    out = out.transpose(1, 2).contiguous().view(batch_sz, seqlen, -1)
    return self.wo(out)

class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.attention = Attention(config)
    self.feed_forward = MLP(config)
    self.attention_norm = RMSNorm(config.embd, config.norm_eps)
    self.ffn_norm = RMSNorm(config.embd, config.norm_eps)

  def forward(self, x, start_pos, freqs_cis, mask):
     x = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
     x = x + self.feed_forward(self.ffn_norm(x))
     return x

class Neo(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.tok_embeddings = nn.Embedding(config.vocab_size, config.embd)
    self.layers = nn.ModuleList()
    for _ in range(config.n_layers):
      self.layers.append(Block(config))
    self.norm = RMSNorm(config.embd, config.norm_eps)
    self.output = nn.Linear(config.embd, config.vocab_size, bias=False)
    self.freq_cis = precompute_freqs_cis(config.head_dim, config.seq_len * 2, config.theta)

  @torch.inference_mode()
  def forward(self, tokens, start_pos):
    bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)
    self.freqs_cis = self.freqs_cis.to(tokens.device)
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
    mask = None
    if seqlen > 1:
      mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

      mask = torch.triu(mask, diagonal=1).to(tokens.device)

    for layer in self.layers:
      h = layer(h, start_pos, freqs_cis, mask)
    h = self.norm(h)
    out = self.output(h).float()
    return out

@dataclass
class ModelConfig:
  embd: int = 4096
  hidden_dim: int = 14336
  n_heads: int = 32
  head_dim: int = embd // n_heads
  n_kv_heads: int = 8
  vocab_size: int = 128256
  n_layers: int = 32
  norm_eps: int = 1e-5
  theta: int = 500000
  seq_len: int = 128
  n_kv_heads_rep: int = n_heads // n_kv_heads

