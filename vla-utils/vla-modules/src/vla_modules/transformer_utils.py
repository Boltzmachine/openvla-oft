import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from time import time


def get_activation(activation="gelu"):
    return {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }[activation]
    
    
class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.,
        drop_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.nolora_fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.nolora_act = act_layer()
        self.nolora_drop1 = nn.Dropout(drop)
        self.nolora_norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.nolora_fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.nolora_drop2 = nn.Dropout(drop) if drop_last else nn.Identity()

    def forward(self, x):
        x = self.nolora_fc1(x)
        x = self.nolora_act(x)
        x = self.nolora_drop1(x)
        x = self.nolora_norm(x)
        x = self.nolora_fc2(x)
        x = self.nolora_drop2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        d_model, 
        n_heads=1,
        proj_drop=0.,
        attn_drop=0.,
        use_torch_attn=True,
    ):
        super(Attention, self).__init__()

        self.d_head, self.n_heads = d_model // n_heads, n_heads
        self.scale = self.d_head ** -0.5

        self.nolora_layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),
        )
        self.nolora_W_output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(proj_drop),
        )
        self.nolora_attn_dropout = nn.Dropout(attn_drop)
        self.use_torch_attn = use_torch_attn

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor=None) -> torch.Tensor:
        # pad_mask: [B, n_heads, N, N], True for padding

        B, N, C = x.shape
        q, k, v = self.nolora_layernorm_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.n_heads), (q, k, v))

        # if pad_mask is None:
        #     pad_mask = torch.zeros((B, self.n_heads, N, N), dtype=torch.bool, device=x.device)

        if self.use_torch_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=~pad_mask if pad_mask is not None else None,
                dropout_p=self.nolora_attn_dropout.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if pad_mask is not None:
                attn.masked_fill_(pad_mask, -1e9)
            attn = attn.softmax(dim=-1)
            attn = self.nolora_attn_dropout(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.nolora_W_output(x)
    
class CrossAttention(nn.Module):
    def __init__(            
        self,
        d_model,
        d_model_q,
        d_model_kv,
        n_heads=1,
        proj_drop=0.,
        attn_drop=0.,
        use_torch_attn=True,
    ):
        super().__init__()

        self.d_head, self.n_heads = d_model // n_heads, n_heads
        self.scale = self.d_head ** -0.5

        self.nolora_layernorm_q = nn.Sequential(
            nn.LayerNorm(d_model_q),
            nn.Linear(d_model_q, d_model)
        )
        self.nolora_layernorm_kv = nn.Sequential(
            nn.LayerNorm(d_model_kv),
            nn.Linear(d_model_kv, d_model * 2)
        )
        self.nolora_W_output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(proj_drop),
        )
        self.nolora_attn_dropout = nn.Dropout(attn_drop)
        self.use_torch_attn = use_torch_attn

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pad_mask: torch.Tensor=None) -> torch.Tensor:
        # pad_mask: [B, n_heads, N, N], True for padding

        B, N, C = q.shape
        q = self.nolora_layernorm_q(q)
        k, v = self.nolora_layernorm_kv(k).chunk(2, dim=-1)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.n_heads), (q, k, v))

        # if pad_mask is None:
        #     pad_mask = torch.zeros((B, self.n_heads, N, N), dtype=torch.bool, device=x.device)

        if self.use_torch_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=~pad_mask if pad_mask is not None else None,
                dropout_p=self.nolora_attn_dropout.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if pad_mask is not None:
                attn.masked_fill_(pad_mask, -1e9)
            attn = attn.softmax(dim=-1)
            attn = self.nolora_attn_dropout(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, -1)
        return self.nolora_W_output(x)


class TransformerBlock(nn.Module):
    def __init__(            
            self,
            d_model,
            n_heads=1,
            activation="gelu",
            attn_drop=0.,
            proj_drop=0.,
            mlp_ratio=4.0,
        ):
        super(TransformerBlock, self).__init__()

        self.attn = Attention(
            d_model=d_model, n_heads=n_heads, proj_drop=proj_drop, attn_drop=attn_drop
        )

        self.mlp = MLP(
            in_features=d_model, hidden_features=int(d_model * mlp_ratio), out_features=d_model, drop=proj_drop, norm_layer=nn.LayerNorm
        )

    def forward(self, token_embs, padding_mask=None):
        context_token_embs = self.attn(token_embs, padding_mask)
        token_embs = token_embs + context_token_embs

        token_embs = token_embs + self.mlp(token_embs)
        return token_embs
    

class Transformer(nn.Module):
    def __init__(            
            self,
            d_model,
            n_layers=1,
            n_heads=1,
            activation="gelu",
            attn_drop=0.,
            proj_drop=0.,
            mlp_ratio=4.0,
        ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                activation=activation,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_ratio=mlp_ratio,
                ) for _ in range(n_layers)])
        
    def forward(self, x, padding_mask=None):
        for block in self.blocks:
            x = block(x, padding_mask)
        return x