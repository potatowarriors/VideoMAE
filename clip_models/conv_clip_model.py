from collections import OrderedDict
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, trunc_normal_
from einops import rearrange




def split_cls(tensor, current_frame):
    cls_tok, pat_tok = tensor[:, :1, :], tensor[:, 1:, :]
    return cls_tok, pat_tok
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class PatchEmbed2D(nn.Module):

    def __init__(
        self,
        patch_size: Tuple[int, int] = (16, 16),
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels

        self.proj = nn.Linear(np.prod(patch_size) * in_channels, embed_dim, bias=False)


    def _initialize_weights(self, x):
        nn.init.kaiming_normal_(self.proj.weight, 0.)
        nn.init.constant_(self.proj.bias, 0.)


    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        pH, pW = self.patch_size

        assert C == self.in_channels and H % pH == 0 and W % pW == 0

        x = x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 1, 3, 5).flatten(3).flatten(1, 2)
        x = self.proj(x)
        
        return x

class ReduceTemporalLayer(nn.Module):
    def __init__(self, num_frame, embed_dim=768, tubelet_size=4):
        super().__init__()
        self.num_frame = num_frame
        self.act = QuickGELU()
        self.downample = nn.Linear(embed_dim, embed_dim//2)
        self.reduce = nn.Conv1d(embed_dim//2, embed_dim//2, kernel_size=3, stride=1, padding=1, groups=embed_dim//2)
        self.upsample = nn.Linear(embed_dim//2, embed_dim)
        
    def forward(self, x):
        cls_tok, pat_tok = split_cls(x, self.num_frame)
        _, n, _ = pat_tok.size()
        pat_tok = self.downample(pat_tok)
        pat_tok = rearrange(pat_tok, '(b t) n d -> (b n) d t', t=self.num_frame)
        pat_tok = self.reduce(pat_tok)
        pat_tok = rearrange(pat_tok, '(b n) d t -> (b t) n d', n = n)
        pat_tok = self.upsample(self.act(pat_tok))
        x = torch.cat((cls_tok, pat_tok), dim=1)
        return x

class Attention(nn.Module):
    '''
    A generalized attention module with more flexibility.
    '''

    def __init__(
        self, q_in_dim: int, k_in_dim: int, v_in_dim: int,
        qk_proj_dim: int, v_proj_dim: int, num_heads: int, out_dim: int,
    ):
        super().__init__()

        self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.out_proj = nn.Linear(v_proj_dim, out_dim)

        self.num_heads = num_heads
        assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        N = q.size(0); assert k.size(0) == N and v.size(0) == N
        Lq, Lkv = q.size(1), k.size(1); assert v.size(1) == Lkv

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        
        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H

        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)

        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk ** 0.5), k)
        aff = aff.softmax(dim=-2)
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        out = self.out_proj(mix.flatten(-2))

        return out



class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        frame_num,
        layer_num,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
        act: nn.Module = QuickGELU,
    ):
        super().__init__()
        
        self.layer_num = layer_num
        self.current_frame = None
        self.reduce = ReduceTemporalLayer(frame_num)

        self.norm1 = nn.LayerNorm(in_feature_dim)
        self.attn = Attention(
            q_in_dim=in_feature_dim, k_in_dim=in_feature_dim, v_in_dim=in_feature_dim,
            qk_proj_dim=qkv_dim, v_proj_dim=qkv_dim, num_heads=num_heads, out_dim=in_feature_dim,
        )

        mlp_dim = round(mlp_factor * in_feature_dim)
        self.norm2 = nn.LayerNorm(in_feature_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
            ('act', act()),
            ('dropout', nn.Dropout(mlp_dropout)),
            ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
        ]))


        self._initialize_weights()


    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)


    def forward(self, x: torch.Tensor):
        x = x + self.reduce(x)
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer2D(nn.Module):

    def __init__(
        self,
        frame_num,
        feature_dim: int = 768,
        input_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_factor: float = 4.0,
        act: nn.Module = QuickGELU,
        ln_pre: bool = True,
    ):
        super().__init__()
        self.frame_num = frame_num

        
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, embed_dim=feature_dim)
        self.num_patches = np.prod([x // y for x, y in zip(input_size, patch_size)]) + 1

        self.cls_token = nn.Parameter(torch.zeros([feature_dim]))
        self.pos_embed = nn.Parameter(torch.zeros([self.num_patches, feature_dim]))
        if ln_pre:
            self.ln_pre = nn.LayerNorm(feature_dim)
        else:
            self.ln_pre = nn.Identity()

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                frame_num, layer_num=i, in_feature_dim=feature_dim, qkv_dim=feature_dim, num_heads=num_heads, mlp_factor=mlp_factor, act=act,
            ) for i in range(num_layers)
        ])
        
        self.reduce_post = ReduceTemporalLayer(2)
        
        self.ln_post = nn.LayerNorm(feature_dim)

        self._initialize_weights()


    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def get_num_layers(self):
        return len(self.blocks)
    
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.size()
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        
        dtype = self.patch_embed.proj.weight.dtype
        x = x.to(dtype)

        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.view(1, 1, -1).repeat(x.size(0), 1, 1), x], dim=1)
        x = x + self.pos_embed

        x = self.ln_pre(x)
        
        for blk in self.blocks:
            x = blk(x)
        x = rearrange(x, '(b t) n d -> b t n d', t=self.frame_num)
        x = x.mean(dim=1) #temporal mean pooling
        
        x = self.ln_post(x[:, 0, :])
        
        return x

class CLIP(nn.Module):
    def __init__(self,
                 image_resolution: int,
                 num_layers: int,
                 feature_dim: int,
                 patch_size: int,
                 num_classes,
                 drop_path,
                 num_frames,
                 batch_size
                 ):
        super().__init__()

        self.layers = num_layers
        self.drop_path_rate = drop_path
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.visual = VisionTransformer2D(
            self.num_frames,
            feature_dim=768,
            input_size=(224,224),
            patch_size=(16,16),
            num_heads=12,
            num_layers=12,
            mlp_factor=4.0,
            act=QuickGELU,
            ln_pre=True
            )
        
        
        self.head = nn.Linear(feature_dim, num_classes)

        self.apply(self.initialize_parameters)
        
    def get_num_layers(self):
        return self.layers
    
    def no_weight_decay(self):
        return {}

    def initialize_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, x):
        return self.visual(x)

    def forward(self, x):
        x = self.encode_image(x)
        x = self.head(x)
        
        return x
