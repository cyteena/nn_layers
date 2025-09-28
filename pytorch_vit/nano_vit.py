# based on simple-vit from lucidrain
from functools import partial

import torch
from torch.nn import Module
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0


# Q: any reason to do pe like these?, or how can I understand the sin cos here?
def posemb_sincos_2d(h, w, dim, temperature: int = 10_000, dtype = torch.float64, device = torch.device('cuda')):
    arange = partial(torch.arange, dtype=dtype, device=device)
    
    # row and col
    y, x = torch.meshgrid(arange(h), arange(w), indexing='ij')

    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'

    omega = arange(dim // 4) / ( dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :] # (H * W, 1) * (1, D/4)
    x = x.flatten()[:, None] * omega[None, :] # (H * W, 1) * (1, D/4)

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    
    return pe

def FeedForward(dim, hidden_dim):
    return nn.Sequential(
        nn.LayerNorm(dim, bias=False),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim)
    )


class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()

        self.norm =  nn.LayerNorm(dim, bias = False)

        self.heads = heads
        self.dim_head = dim_head

        dim_inner = heads * dim_head

        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_k = nn.Linear(dim, dim_inner, bias = False)
        self.to_v = nn.Linear(dim, dim_inner, bias = False)

        self.q_norm = nn.LayerNorm(dim_head, bias = False)
        self.k_norm = nn.LayerNorm(dim_head, bias = False)

        self.out = nn.Linear(dim_inner, dim, bias = False)


    def forward(self, x, context: Tensor | None = None):

        x = self.norm(x)

        context = default(context, x)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        def split_heads(t: Tensor):
            return t.unflatten(-1, (self.heads, self.dim_head))
        
        def transpose_head_seq(t: Tensor):
            return t.transpose(1, 2)
        
        q, k, v = map(split_heads, (q, k, v))

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = map(transpose_head_seq, (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v)

        out = transpose_head_seq(out).flatten(-2)

        return self.out(out)
    

class TransformerBlock(Module):
    def __init__(self, dim, heads, dim_head, mlp_dim) -> None:
        super().__init__()
        self.attention = Attention(dim, heads, dim_head)
        self.feed_forward = FeedForward(dim, mlp_dim)

    def forward(self, x, context: Tensor | None = None):
        x = x + self.attention(x, context)
        x = x + self.feed_forward(x)
        return x
    

class Transformer(Module):
    def __init__(self, depth, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim))
    
    def forward(self, x, context: Tensor | None = None):
        for layer in self.layers:
            x = layer(x, context)
        return x

class ImageEmbedder(Module):
    def __init__(self, image_size, patch_size, dim, channel) -> None:
        super().__init__()
        
        img_height, img_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert divisible_by(img_height, patch_height) and divisible_by(img_width, patch_width), 'Image dimensions must be divisible by the patch size.'
        
        self.num_patches = (img_height // patch_height) * (img_width // patch_width)
        self.to_patches = Rearrange('... c (h p1) (w p2) -> ... h w (c p1 p2)', p1=patch_height, p2=patch_width)

        
        self.patch_to_embed = nn.Sequential(
            nn.LayerNorm(channel * patch_height * patch_width, bias = False),
            nn.Linear(channel * patch_height * patch_width, dim, bias= False),
            nn.LayerNorm(dim, bias = False)
        )

    def forward(self, img: Tensor):
        patches = self.to_patches(img) # (h, w, patch_dim)
        dim_patch_height, dim_patch_width, _ = patches.shape[-3:]
        flatten_patches = rearrange(patches, '... h w d -> ... (h w) d')
        tokens = self.patch_to_embed(flatten_patches)

        tokens += posemb_sincos_2d(dim_patch_height, dim_patch_width, tokens.shape[-1], device = tokens.device, dtype = tokens.dtype)

        print(f"{tokens.shape=}, is the first dim should be batch size?")
        return tokens

class SimpleViT(Module):

    def __init__(
            self,
            *,
            img_size,
            patch_size,
            num_classes,
            depth,
            dim,
            heads,
            dim_head,
            mlp_dim,
            channels =3
    ):
        super().__init__()

        self.img_embedder = ImageEmbedder(img_size, patch_size, dim, channels)
        self.transformer = Transformer(depth, dim, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, img: Tensor):

        assert img.ndim == 4 and img.shape[1] == 3, 'img must be of shape (batch, channels, height, width)'

        tokens = self.img_embedder(img) # (batch, n_patches, dim)

        x = self.transformer(tokens)

        x = x.mean(dim = 1)

        x = self.to_latent(x)

        return self.linear_head(x)
    



if __name__ == "__main__":

    # test SimpleViT
    model = SimpleViT(
        img_size = 224,
        patch_size = 16,
        num_classes = 1000,
        depth = 6,
        dim = 512,
        heads = 8,
        dim_head = 64,
        mlp_dim = 1024,
        channels = 3
    )

    imgs = torch.randn(2, 3, 224, 224)
    logits = model(imgs)
    assert logits.shape == (2, 1000)
    print(logits.shape)  # should be [2, 1000]
