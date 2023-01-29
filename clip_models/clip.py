import os
import warnings
from typing import Union
from pkg_resources import packaging
from importlib import import_module
import sys

import torch
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize"]


def load(model_path, args, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
    if os.path.isfile(model_path):
        weight_path = model_path
    else:
        raise RuntimeError(f"Model {model_path} not found; check the model name")
    
    with open(weight_path, 'rb') as opened_file:
        model = torch.jit.load(opened_file, map_location='cpu')
        state_dict = None
        
    if args.clip_model == 't2s':
        from .t2s_clip_model import build_model
    elif args.clip_model == 'conv':
        from .attnpool_clip_model import build_model
    elif args.clip_model == 'attnpool':
        from .attnpool_clip_model import build_model
    else:
        raise ImportError
    
    model = build_model(state_dict or model.state_dict(), args).to(device)
    if str(device) == "cpu":
        model.float()
    return model
