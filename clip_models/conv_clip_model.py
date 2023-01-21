from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from timm.models.layers import drop_path, trunc_normal_




class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        if orig_type == torch.float16:
            ret = super().forward(x)
        elif orig_type == torch.float32:
            ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
    
class CrossAttentionT2S(nn.Module): # 이게 VMAE로 치면 blocks class다. 여기에 cross s2t_attn layer가 추가되어야 한다.
    def __init__(self, dim: int, n_head: int, num_frames, attn_mask: torch.Tensor = None):
        super().__init__()

        # add for cross-attn
        self.num_head = n_head
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        self.num_frames = num_frames
        
        #여기에 cross attn t2s module이 들어가야 한다.
        self.t2s_q = nn.Linear(dim, all_head_dim, bias=False)
        self.t2s_q_bias = nn.Parameter(torch.zeros(dim))
        
        self.t2s_kv = nn.Linear(dim, all_head_dim * 2, bias=False)
        self.t2s_kv_bias = nn.Parameter(torch.zeros(dim))
        
        self.t2s_proj = nn.Linear(all_head_dim, dim)
        # 여기에 drop out 할지 말지는 고민좀 해보자.
        
        self.attn_mask = attn_mask
    
    def t2s_cross_attn(self, s_x, t_x):
        s_x = rearrange(s_x, 'p b c -> b p c')
        B, t_N, C = t_x.shape
        s_x = rearrange(s_x, '(b t) p d -> b (t p) d', b=B)
        _, s_N, C = s_x.shape
        t2s_q_bias = self.t2s_q_bias
        t2s_kv_bias = torch.cat((torch.zeros_like(self.t2s_kv_bias, requires_grad=False), self.t2s_kv_bias))
        
        t2s_q = F.linear(input=s_x, weight=self.t2s_q.weight, bias=t2s_q_bias)
        t2s_q = t2s_q.reshape(B, s_N, self.num_head, -1).permute(0, 2, 1, 3)
        t2s_kv = F.linear(input=t_x.half(), weight=self.t2s_kv.weight, bias=t2s_kv_bias)
        t2s_kv = t2s_kv.reshape(B, t_N, 2, self.num_head, -1).permute(2, 0, 3, 1, 4)
        t2s_q, t2s_k, t2s_v = t2s_q, t2s_kv[0], t2s_kv[1]
        
        t2s_q = t2s_q * self.scale
        t2s_attn = (t2s_q @ t2s_k.transpose(-2, -1))
        
        t2s_attn = t2s_attn.softmax(dim=-1)
        
        s_x = (t2s_attn @ t2s_v).transpose(1, 2).reshape(B, s_N, -1)
        s_x = self.t2s_proj(s_x)
        s_x = rearrange(s_x, 'b (t p) d -> (b t) p d', t =self.num_frames)
        s_x = rearrange(s_x, 'b s c -> s b c')
        
        return s_x

    def forward(self, s_x: torch.Tensor, t_x: torch.Tensor):
        return self.t2s_cross_attn(s_x, t_x)
    
class ReduceTemporalLayer(nn.Module):
    def __init__(self, current_frame, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_num = img_size // patch_size
        self.chans = in_chans
        self.current_frame = current_frame
        self.reduce = nn.Conv1d(embed_dim, embed_dim, kernel_size=tubelet_size, stride=2, groups=embed_dim)
        
    def forward(self, x):
        t = self.current_frame # reduce된 frame수
        b = x.shape[1] // t # frame 수 기준 batch size 계산
        x = rearrange(x, 'n (b t) d -> b t n d', b=b)
        B, T, N, D = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().flatten(0, 1) # B * T, N, D
        x = self.reduce(x)
        x = x.view(B, N, D, -1).permute(0, 3, 1, 2).contiguous() # B, T, N, D
        x = rearrange(x, 'B T N D -> N (B T) D')
        
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, drop_path_rate, num_frames, layer_num, batch_size, attn_mask: torch.Tensor = None):
        super().__init__()

        self.layer_num = layer_num
        if self.layer_num == 0:
            self.reduce = nn.Identity()
        elif self.layer_num % 3 == 0:
            current_frame = num_frames // (2**(self.layer_num//3 - 1))
            self.reduce = ReduceTemporalLayer(current_frame)
        else:
            self.reduce = nn.Identity()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = self.reduce(x)
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, drop_path_rate, num_frames, batch_size, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.layers)]
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, dpr[i], num_frames, i, batch_size, attn_mask) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        for blk in self.resblocks:
            x = blk(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate: float, num_frames: int, batch_size: int):
        super().__init__()
        self.layers = layers
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, drop_path_rate, num_frames, batch_size)
        self.reduce_post = ReduceTemporalLayer(current_frame=2)

        self.ln_post = LayerNorm(width)
        
        

    def forward(self, x: torch.Tensor):
        x = rearrange(x, 'b c t h w -> (b t) c h w') # for independently extract frame feature
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = self.reduce_post(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        return x


class CLIP(nn.Module):
    def __init__(self,
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 num_classes,
                 drop_path,
                 num_frames,
                 batch_size
                 ):
        super().__init__()

        vision_heads = vision_width // 64
        self.layers = vision_layers
        self.drop_path_rate = drop_path
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            drop_path_rate=self.drop_path_rate,
            num_frames = self.num_frames,
            batch_size=self.batch_size
            )
        
        
        self.head = nn.Linear(vision_width, num_classes)

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
            

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, x):
        return self.visual(x)

    def forward(self, x):
        x = self.encode_image(x)
        x = self.head(x)
        
        return x


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, args):
    
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    num_frames = args.num_frames
    num_classes = args.nb_classes
    drop_path = args.drop_path
    batch_size = args.batch_size


    model = CLIP(
        image_resolution, vision_layers, vision_width, vision_patch_size, num_classes, drop_path, num_frames, batch_size
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model