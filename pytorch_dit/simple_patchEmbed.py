# implement PatchEmbed for Dit, it's just use Conv2d, then you can flatten and transpose

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from einops.layers.torch import Rearrange

from typing import Optional, Callable

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

class PatchEmbed(Module):
    """
    Use Conv2d to patchify the image, and then we flatten them
    Conv2d then you can normalize them.
    """

    def __init__(self, img_size, patch_size, in_channels, embed_dim, post_norm_layer: Optional[Callable[[Tensor]]] = None) -> None:
        super().__init__()
        img_height, img_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        assert img_height % patch_height == 0 and img_width % patch_width == 0, "Image dimensions must be divisible by the patch size."

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = post_norm_layer(embed_dim) if post_norm_layer else nn.Identity()
        self.to_tokens = Rearrange('... c h w -> ... (h w) c') # we use this layer, after the conv

    
    def forward(self, img: Tensor) -> Tensor:
        assert img.dim() == 4, "Input image must be 4D tensor"
        x = self.proj(img)  # (B, embed_dim, H/patch_height, W/patch_width)
        x = self.norm(x)
        x = self.to_tokens(x)  # (B, num_patches, embed_dim)
        return x

