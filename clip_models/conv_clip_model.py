from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from timm.models.layers import drop_path, trunc_normal_



def cls_split(tensor):
    # tensor (b t) n d
    cls_tok, pat_tok = tensor[:1, :, :], tensor[1:, :, :]
    return cls_tok, pat_tok

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
    
    
class ReduceTemporalLayer(nn.Module):
    def __init__(self, current_frame, cls_split, kernel_size, stride, pad_size, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=3):
        super().__init__()
        self.current_frame = current_frame
        self.cls_split = cls_split
        self.img_size = img_size
        self.num_frames = num_frames
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if self.cls_split:
            self.patch_num = (img_size // patch_size) ** 2 #cls token +1
        else:
            self.patch_num = (img_size // patch_size) ** 2 + 1 #cls token +1
        self.act = QuickGELU()
        self.max_pool = nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.down = nn.Linear(self.embed_dim, self.embed_dim//2)
        self.reduce = nn.Conv1d(self.embed_dim//2, self.embed_dim//2, kernel_size=(kernel_size[0]), stride=(stride[0]), padding=(pad_size[0]), groups=self.embed_dim//2)
        self.up = nn.Linear(self.embed_dim//2, self.embed_dim)
        self.temporal_posembed = nn.Parameter(torch.zeros(self.current_frame//2, self.embed_dim))
        
    def forward(self, x):
        if self.cls_split:
            cls_tok, x = cls_split(x) # x is patch token
        x = self.down(x)
        x = rearrange(x, 'n (b t) d -> (b n) d t', t=self.current_frame)
        x = self.reduce(x)
        x = rearrange(x, '(b n) d t -> n (b t) d', n=self.patch_num)
        x = self.up(self.act(x))
        x = rearrange(x,'n (b t) d -> n b t d', t=self.current_frame//2)
        x = x + self.temporal_posembed.to(x.dtype).view(1, 1, self.current_frame//2, self.embed_dim)
        x = rearrange(x, 'n b t d -> n (b t) d')
        if self.cls_split:
            cls_tok = rearrange(cls_tok, 'n (b t) d -> (b n) d t', t=self.current_frame)
            cls_tok = self.max_pool(cls_tok)
            cls_tok = rearrange(cls_tok, 'b d t -> (b t) d').unsqueeze(dim=0)
            x = torch.cat((cls_tok, x), dim = 0)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, drop_path_rate, num_frames, layer_num, batch_size, cls_split, 
                 reduce_position, kernel_size, stride, pad_size, attn_mask: torch.Tensor = None):
        super().__init__()

        self.layer_num = layer_num
        self.current_frame = None
        self.dim = d_model
        if self.layer_num not in reduce_position:
            pass
        else:
            self.current_frame = num_frames // (2 ** (reduce_position.index(layer_num)))
            self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding= 0)
            self.reduce = ReduceTemporalLayer(self.current_frame, cls_split, kernel_size, stride, pad_size)
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
        if self.current_frame is not None:
            b = x.shape[1] // self.current_frame
            x_half = rearrange(x, 'n (b t) d -> (b n) d t', t=self.current_frame)
            x_half = self.max_pool(x_half)
            x_half = rearrange(x_half, '(b n) d t -> n (b t) d', b=b)
            x = x_half + self.drop_path(self.reduce(x))
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, drop_path_rate, num_frames, batch_size, cls_split, 
                 reduce_position, kernel_size:list, strdie:list, pad_size:list, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.layers)]
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, dpr[i], num_frames, i, batch_size, cls_split, reduce_position, kernel_size, strdie, pad_size, attn_mask) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        for blk in self.resblocks:
            x = blk(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate: float, num_frames: int, batch_size: int, cls_split:bool,
                 reduce_position:list, kernel_size:list, stride:list, pad_size:list):
        super().__init__()
        self.layers = layers
        self.input_resolution = input_resolution
        self.embed_dim = width
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.temporal_posembed = nn.Parameter(torch.zeros([num_frames, width]))

        self.transformer = Transformer(width, layers, heads, drop_path_rate, num_frames, batch_size, cls_split, reduce_position, kernel_size, stride, pad_size)

        self.ln_post = LayerNorm(width)
        
    def get_num_layers(self):
        return self.layers

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        if len(x.size()) == 5:
            b, c, t, h, w = x.size()
            all_frame_setting = True
            x = rearrange(x, 'b c t h w -> (b t) c h w') # for independently extract frame feature
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        
        x = self.ln_pre(x)
        if all_frame_setting:
            x = rearrange(x, '(b t) n d -> b t n d', b=b)
            x = x + self.temporal_posembed.to(x.dtype).view(1, t, 1, self.embed_dim)
            x = rearrange(x, 'b t n d -> (b t) n d')
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
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
                 batch_size,
                 cls_split : bool,
                 reduce_position :list,
                 kernel_size : list,
                 stride : list,
                 pad_size : list,
                 
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
            batch_size=self.batch_size,
            cls_split = cls_split,
            reduce_position = reduce_position,
            kernel_size = kernel_size,
            stride = stride,
            pad_size = pad_size
            )
        
        
        self.head = nn.Linear(vision_width, num_classes)

        self.apply(self.initialize_parameters)
        
    def get_num_layers(self):
        return self.layers
    
    def no_weight_decay(self):
        return {'visual.class_embedding', 'visual.conv1.weight'}

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
    cls_split = args.cls_split
    reduce_position = args.reduce_position
    kernel_size = args.kernel_size
    stride = args.stride
    pad_size = args.pad_size


    model = CLIP(
        image_resolution, vision_layers, vision_width, vision_patch_size, num_classes, drop_path, num_frames, batch_size, cls_split,
        reduce_position, kernel_size, stride, pad_size
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model