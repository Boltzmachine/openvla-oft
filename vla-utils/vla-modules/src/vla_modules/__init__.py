import torch
from torch import nn
from vit_pytorch.vit import Attention, FeedForward, Transformer2
from vector_quantize_pytorch import VectorQuantize, FSQ
from einops import rearrange


class QueryTransformer(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.q = nn.Parameter(torch.randn(1, 256, dim))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, additional_q):
        q = self.q + additional_q
        x = self.norm(x)

        kv = self.to_kv(x).chunk(2, dim = -1)
        qkv = tuple([q, *kv])
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
          

class AttentionPooling(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = Attention(dim, heads=heads, dim_head=dim // heads, dropout=0.1)

    def forward(self, x):
        x = torch.cat([self.cls_embedding.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.attn(x)
        return x[:, 0, :]  # (B, dim)
      
                
class DisentangleAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 2304,
        n_token: int = 256,
        static_ratio: float = 0.5,
        backbone: str = "transformer",
        quantizer: str = "fsq",
    ):
        super().__init__()
        
        self.pos_embedding = nn.Parameter(torch.randn(1, n_token + 1, hidden_dim))
        self.type_embedding = nn.Embedding(2, hidden_dim)
        self.backbone = backbone
        if backbone == "transformer":
            self.common_layers = Transformer2(hidden_dim, depth=1, dim_head=hidden_dim // 8, heads=8, mlp_dim=hidden_dim, dropout=0.1)
        elif backbone == "mlp":
            self.common_layers = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif backbone == "query_transformer":
            self.common_layers = QueryTransformer(hidden_dim, heads=8, dim_head=hidden_dim // 8, dropout=0.1)
        else:
            raise
        
        static_dim = int(n_token * static_ratio)  # static dim
        dynamic_dim = n_token - static_dim  # dynamic dim
        self.register_buffer("static_dim", torch.tensor(static_dim, dtype=torch.int64))
        self.register_buffer("dynamic_dim", torch.tensor(dynamic_dim, dtype=torch.int64))
        
        self.quantizer = quantizer
        if quantizer == "vq":
            self.vq = VectorQuantize(
                    dim = hidden_dim,
                    codebook_size = 512,     # codebook size
                    decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
                    commitment_weight = 1.,   # the weight on the commitment loss
                    use_cosine_sim=False,
                )
        elif quantizer == "fsq":
            self.fsq = FSQ(
                levels = [8, 5, 5, 5, 5],
                dim=hidden_dim,
            )
        elif quantizer == "vq_cos":
            self.vq = VectorQuantize(
                    dim = hidden_dim,
                    codebook_size = 512,     # codebook size
                    decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
                    commitment_weight = 1.,   # the weight on the commitment loss
                    use_cosine_sim=True,
                )
        elif quantizer == "none":
            self.vq = nn.Identity()
        else:
            raise ValueError(f"Unknown quantizer: {quantizer}. Supported: ['vq', 'fsq', 'none']")
        
        if quantizer in ['vq_cos', 'fsq']:
            self.post_vq = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.post_vq = nn.Identity()

    def forward(self, features):
        type_id = torch.tensor([0] * self.static_dim + [1] * self.dynamic_dim, device=features.device).unsqueeze(0)  # (1, C)
        type_embedding = self.type_embedding(type_id)  # (1, C, D)
        if self.backbone == "query_transformer":
            type_embedding = type_embedding.expand(features.shape[0], -1, -1)  # (B, C, D)
            features = self.common_layers(features, type_embedding)
        else:
            # features = features + type_embedding  # (B, C, D)
            features = self.common_layers(features)
        static_features = features[:, :self.static_dim]
        dynamic_features = features[:, self.static_dim:]
        
        # features = features + self.pos_embedding[:, :features.shape[1], :]
        # static_features = self.static_layers(features).reshape(features.size(0), self.static_dim, -1)  # (B, C, D)
        # dynamic_features = self.dynamic_layers(features).reshape(features.size(0), self.dynamic_dim, -1)  # (B, C, D)
        if self.quantizer == "none":
            return {"features": static_features}, dynamic_features
        elif self.quantizer == "vq" or self.quantizer == "vq_cos":
            quantized, indices, commit_loss, dists = self.vq(static_features, return_distances=True) # (B, C, D), (B, C), (B)
            quantized, commit_loss = quantized.to(features.dtype), commit_loss.to(features.dtype)
            logprobs = dists.squeeze(0).log_softmax(-1)
            # probs = dists.squeeze(0).softmax(-1)
            return { 'features': self.post_vq(quantized), 'indices': indices, 'commit_loss': commit_loss, 'logprobs': logprobs, 'dists': dists.squeeze(0), "features_cont": static_features }, dynamic_features
        elif self.quantizer == "fsq":
            quantized, indices, dists = self.fsq(static_features, return_distances=True)
            quantized = self.post_vq(quantized)
            # commit_loss = torch.nn.functional.l1_loss(quantized, static_features)
            return { 'features': quantized, "indices": indices.long(), "raw": static_features }, dynamic_features
            return { 'features': quantized, "indices": indices.long(), 'dists': dists, "raw": static_features }, dynamic_features
        else:
            raise ValueError(f"Unknown quantizer: {self.quantizer}. Supported: ['vq', 'fsq', 'none']")
    

class ActionPredictor(nn.Module):
    def __init__(self, dim, action_tokenizer, mode='regression'):
        super().__init__()
        assert mode in ['classification', 'regression'], "mode must be either 'classification' or 'regression'"
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.attn_vertical = Attention(dim*2, heads = 8, dim_head = dim // 8, dropout = 0.1)
        # self.attn = Attention(dim, heads = 8, dim_head = dim // 8, dropout = 0.1)
        # self.ff = FeedForward(dim, dim, dropout = 0.1)
        self.attn = Transformer2(dim, depth=1, dim_head=dim // 8, heads=8, mlp_dim=dim, dropout=0.1)
        translation_dim = (action_tokenizer.translation_tokenizer.token_end_idx - action_tokenizer.translation_tokenizer.token_start_idx + 1) if mode == 'classification' else 3
        rotation_dim = (action_tokenizer.rotation_tokenizer.token_end_idx - action_tokenizer.rotation_tokenizer.token_start_idx + 1) if mode == 'classification' else 3
        gripper_dim = (action_tokenizer.gripper_tokenizer.token_end_idx - action_tokenizer.gripper_tokenizer.token_start_idx + 1) if mode == 'classification' else 1
        self.head = nn.ModuleDict({
            "translation": nn.Linear(dim, translation_dim),
            "rotation": nn.Linear(dim, rotation_dim),
            "gripper": nn.Linear(dim, gripper_dim),
        })
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, dim))  # position embedding for cls token
        
    def forward(self, x1, x2):
        x1 = x1 + self.pos_embedding[:, 0, :].unsqueeze(1)
        x2 = x2 + self.pos_embedding[:, 1, :].unsqueeze(1)
        if hasattr(self, "attn_vertical"):
            x = self.attn_vertical(torch.cat([x1, x2], dim=-1)) # (B, N, D*2)
            x1, x2 = x.split(x1.shape[-1], dim=-1)  # (B, N, D), (B, N, D)
        x = torch.cat([self.cls_token.expand(x1.shape[0], -1, -1), x1, x2], dim=1)  # (B, N+2, D)
        x = self.attn(x)
        cls_token = x[:, 0, :]  # (B, D)
        return {
            "translation": self.head["translation"](cls_token),
            "rotation": self.head["rotation"](cls_token),
            "gripper": self.head["gripper"](cls_token),
        }