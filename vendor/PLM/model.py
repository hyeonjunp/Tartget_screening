# model.py
import math
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
import uuid


# -------------------------
# utils
# -------------------------
def utils_softmax(x: Tensor, dim: int, onnx_trace: bool = False) -> Tensor:
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def gelu(x: Tensor) -> Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x: Tensor) -> Tensor:
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        mean = x.mean(dims, keepdim=True)
        x0 = x - mean
        var = x0.pow(2).mean(dims, keepdim=True)
        x = x0 / torch.sqrt(var + self.eps)
        if self.affine:
            x = self.weight * x + self.bias
        return x


# -------------------------
# Fairseq incremental state (keep, but not used in your training)
# -------------------------
class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(
        b for b in cls.__bases__ if b != FairseqIncrementalState
    )
    return cls

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, padding_idx: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.weights = None  # (num_positions, embed_dim)

    def forward(self, position_ids: torch.Tensor):
        """
        position_ids: (B, T), FloatTensor (can be non-integers)
        """
        # (B, T) -> (B, T, 1)
        # Using the same logic as get_embedding but dynamically for input positions.
        
        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        
        # freqs: (half_dim,)
        freqs = torch.exp(
            torch.arange(half_dim, dtype=torch.float, device=position_ids.device) * -emb_scale
        )
        
        # position_ids: (B, T) -> (B, T, 1)
        # freqs: (half_dim) -> (1, 1, half_dim)
        # argument: (B, T, half_dim)
        args = position_ids.unsqueeze(-1) * freqs.view(1, 1, -1)
        
        # sin, cos -> (B, T, half_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :, :1])], dim=-1)
            
        # pad token handling (if position_ids is exactly 0 for pad, we might want to mask it manually in model)
        # But here we assume position_ids strictly follow distance logic.
        # If necessary, one can mask embedding by `padding_mask` outside.
        
        return embedding

    # get_embedding is removed as it's for fixed integer indices


# -------------------------
# Multi-head attention
# -------------------------
@with_incremental_state
class MultiheadAttention(nn.Module):
    """
    ESM-style Multi-head Attention (RotaryEmbedding 제거 버전)
    input/output: (T, B, C)
    returns:
      attn_out: (T, B, C)
      attn_weights_out:
        - if need_head_weights=True: (H, B, T, S)
        - else if need_weights=True: (B, T, S)  (head 평균)
        - else: None
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = (self.kdim == embed_dim and self.vdim == embed_dim)

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

        self.onnx_trace = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: Tensor,                      # (T,B,C)
        key: Optional[Tensor],              # (S,B,C) or None
        value: Optional[Tensor],            # (S,B,C) or None
        key_padding_mask: Optional[Tensor] = None,  # (B,S) bool, True=PAD
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = False,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,         # (T,S) additive mask (e.g., -inf)
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        # QKV projection
        q = self.q_proj(query)
        k = self.k_proj(query if key is None else key)
        v = self.v_proj(query if value is None else value)

        q = q * self.scaling

        # reshape: (B*H, T, D)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # scores: (B*H, T, S)
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        # additive attn mask
        if attn_mask is not None:
            # attn_mask: (T,S) -> broadcast to (B*H,T,S)
            attn_weights = attn_weights + attn_mask.unsqueeze(0)

        # key padding mask
        if key_padding_mask is not None:
            # (B,S) -> (B,1,1,S)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, -1)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, -1)

        if before_softmax:
            return attn_weights, v

        # softmax (stabilize: when a row is all -inf, softmax => NaN)
        attn_weights_float = utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights_float = torch.nan_to_num(attn_weights_float, nan=0.0)

        attn_probs = F.dropout(attn_weights_float, p=self.dropout, training=self.training)

        # out: (B*H,T,D)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # weights output
        attn_weights_out = None
        if need_weights:
            attn_weights_out = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, -1
            ).type_as(attn).transpose(1, 0)  # (H,B,T,S)

            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(dim=0)  # (B,T,S)

        return attn, attn_weights_out


# -------------------------
# Transformer layer
# -------------------------
class TransformerLayer(nn.Module):
    """
    ESM2-style Transformer block (RoPE 없음)
    input/output: (T, B, C)
    """
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        dropout: float = 0.1,
        add_bias_kv: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.self_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            dropout=dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=False,
            self_attention=True,
        )
        self.self_attn_layer_norm = LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,                                 # (T,B,C)
        self_attn_padding_mask: Optional[Tensor] = None,  # (B,T) bool
        self_attn_mask: Optional[Tensor] = None,          # (T,T)
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        # Self-attention
        residual = x
        x = self.self_attn_layer_norm(x)

        x_attn, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + F.dropout(x_attn, p=self.dropout, training=self.training)

        # FFN
        residual = x
        x = self.final_layer_norm(x)
        x = self.fc2(F.dropout(gelu(self.fc1(x)), p=self.dropout, training=self.training))
        x = residual + F.dropout(x, p=self.dropout, training=self.training)

        return x, attn


# -------------------------
# MLM head
# -------------------------
class PLMHead(nn.Module):
    """Head for masked language modeling."""
    def __init__(self, embed_dim: int, output_dim: int, weight: nn.Parameter):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = LayerNorm(embed_dim)
        self.weight = weight  # weight tying
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features: Tensor) -> Tensor:
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x


# -------------------------
# Protein MLM model
# -------------------------
class ProteinMLM(nn.Module):
    """
    masked LM:
      input_ids: (B, T)
      position_ids: (B, T)
      attention_mask: (B, T) 1=token, 0=pad
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 1024,
        num_layers: int = 32,
        num_heads: int = 32,
        ffn_embed_dim: int = 4096,
        max_position_embeddings: int = 2048,
        padding_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embed_dim = embed_dim

        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # position embedding은 보통 padding_idx 따로 안 둠 (원하면 넣어도 됨)
        #self.embed_positions = nn.Embedding(max_position_embeddings, embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim=embed_dim,
            padding_idx=padding_idx,
        )
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim=embed_dim,
                ffn_embed_dim=ffn_embed_dim,
                attention_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_layer_norm = LayerNorm(embed_dim)
        self.lm_head = PLMHead(embed_dim, vocab_size, self.embed_tokens.weight)

    def forward(
        self,
        input_ids: torch.LongTensor,             # (B,T)
        position_ids: torch.Tensor,              # (B,T) float
        attention_mask: Optional[torch.LongTensor] = None,  # (B,T)
        return_attn: bool = False,
        need_head_weights: bool = False,
    ):
        # embeddings: (B,T,C)
        x = self.embed_tokens(input_ids) + self.embed_positions(position_ids)

        # (B,T,C) -> (T,B,C)
        x = x.transpose(0, 1)

        # padding mask for attention: True where PAD
        self_attn_padding_mask = None
        if attention_mask is not None:
            self_attn_padding_mask = attention_mask.eq(0)

        all_attn: Optional[List[Tensor]] = [] if return_attn else None

        for layer in self.layers:
            x, attn = layer(
                x,
                self_attn_padding_mask=self_attn_padding_mask,
                need_head_weights=need_head_weights,
            )
            if return_attn:
                all_attn.append(attn)

        x = self.final_layer_norm(x)

        # back to (B,T,C)
        x = x.transpose(0, 1)
        logits = self.lm_head(x)  # (B,T,V)

        if return_attn:
            return logits, x,all_attn
        return logits, x
