from __future__ import annotations
from typing import List
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torch.nested import nested_tensor


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


def FeedForward(dim, hidden_dim, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim, bias=False),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, bias = False)

        dim_inner = heads * dim_head
        
        self.heads = heads
        self.dim_head = dim_head
        # maybe we have a scale here?
        self.to_query = nn.Linear(dim, dim_inner, bias=False)
        self.to_key = nn.Linear(dim, dim_inner, bias=False)
        self.to_value = nn.Linear(dim, dim_inner, bias=False)

        # qk norm
        self.q_norm = nn.LayerNorm(dim_head, bias = False)
        self.k_norm = nn.LayerNorm(dim_head, bias = False)


        self.dropout = dropout
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    
    def forward(self, x, context: Tensor | None = None):

        x = self.norm(x) # pre_norm

        context = default(context, x)

        query = self.to_query(x)
        key = self.to_key(context)
        value = self.to_value(context)


        def spilt_heads(t: Tensor):
            return t.unflatten(-1, (self.heads, self.dim_head))
        
        def transpose_head_seq(t: Tensor):
            return t.transpose(1, 2) # (b, heads, n, dim_head)
        
        query, key, value = map(spilt_heads, (query, key, value))

        query = self.q_norm(query)
        key = self.k_norm(key)

        query, key, value = map(transpose_head_seq, (query, key, value))

        out = F.scaled_dot_product_attention(query, key, value, dropout_p= self.dropout if self.training else 0.)

        out = transpose_head_seq(out).flatten(2)

        return self.to_out(out) # let multi-heads talk to each other
    

class TransformerBlock(Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.) -> None:
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x # we don't do norm here, leave residual path alone

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim, dropout))

        self.norm = nn.LayerNorm(dim, bias = False)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x) # we don't use residual here, already done in block, we need just one residual path 

        return x 

class NaViT(Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0. ,
        token_dropout_prob : float | int
    ):
        super().__init__()
        img_height, img_width = pair(image_size)
        
        self.token_dropout_prob = token_dropout_prob

        assert divisible_by(img_height, patch_size) and divisible_by(img_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = img_height // patch_size, img_width // patch_size

        patch_dim = channels * patch_size * patch_size

        self.channels = channels
        self.patch_size = patch_size
        # how can I make sure this will do the right thing?
        self.to_patches = Rearrange('c (h p1) (w p2) -> h w (c p1 p2)', p1 = patch_size, p2 = patch_size)

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim), # if we have normalized the img, we should set bias = False
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # final attention
        self.attn_pool_queries = nn.Parameter(torch.randn(dim)) # one query vector
        self.attn_pool = Attention(dim, dim_head, heads)
        
        # output to logits
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim, bias=False),
            nn.Linear(dim, num_classes, bias=False)
        )
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def patch_embed_dim(self):
        return self.to_patch_embedding[-2].out_features # the linear layer output dim
    
    @property
    def num_patches(self):
        """
        if you use variable length input, this should not be used 
        """
        img_height, img_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)
        return (img_height // patch_height) * (img_width // patch_width)
    
    def forward(
        self,
        imgs: List[Tensor] # we allow different size
    ):
        batch, device = len(imgs), self.device

        arange = partial(torch.arange, device=device)

        assert all([img.ndim == 3 and img.shape[0] == self.channels for img in imgs])

        all_patches = [self.to_patches(img) for img in imgs]

        positions = []

        for patches in all_patches:
            patch_height, patch_width = patches.shape[:2] # not the height of patch, but the dim of patch grid

            # (h, w, 2）
            # I don't know the ij and xy 
            hw_indices = torch.stack(torch.meshgrid(arange(patch_height), arange(patch_width), indexing='ij'), dim = -1)
            hw_indices = rearrange(hw_indices, 'h w c -> (h w) c')

            positions.append(hw_indices)

        
        all_tokens = [rearrange(patches, 'h w d -> (h w) d') for patches in all_patches]

        seq_lens = torch.tensor([i.shape[0] for i in all_tokens], device=device)

        if self.training and self.token_dropout_prob > 0:

            keep_seq_lens = ((1 - self.token_dropout_prob) * seq_lens).int()

            kept_tokens = []
            kept_positions = []

            for one_image_tokens, one_image_posistions, seq_len, num_keep in zip(all_tokens, positions, seq_lens, keep_seq_lens):
                keep_indices = torch.randn(int(seq_len.item()), device=device).topk(int(num_keep.item()), dim = -1).indices

                one_image_kept_tokens = one_image_tokens[keep_indices]
                one_image_kept_positions = one_image_posistions[keep_indices]

                kept_tokens.append(one_image_kept_tokens)
                kept_positions.append(one_image_kept_positions)

            all_tokens, positions, seq_lens = kept_tokens, kept_positions, keep_seq_lens

        
        height_indices, width_indices = torch.cat(positions).unbind(dim = -1)
        height_embed, width_embed = self.pos_embed_height[height_indices], self.pos_embed_width[width_indices]

        pos_embed = height_embed + width_embed

        # cat all tokens, then split 
        tokens = torch.cat(all_tokens)

        tokens = self.to_patch_embedding(tokens)

        # absolute positions

        tokens = tokens + pos_embed

        # what's the meaning of torch.jagged here?
        tokens = nested_tensor(tokens.split(seq_lens.tolist(), dim = 0), layout=torch.jagged)

        tokens = self.dropout(tokens)

        tokens = self.transformer(tokens)

        attn_pool_queries = [rearrange(self.attn_pool_queries, '...->1 ...')] * batch
        attn_pool_queries = nested_tensor(attn_pool_queries, layout=torch.jagged)

        pooled = self.attn_pool(attn_pool_queries, tokens)

        # back to unjagged
        # what's the meaning here?
        
        logits = torch.stack(pooled.unbind())

        logits = rearrange(logits, 'b 1 d -> b d')

        logits = self.to_latent(logits) # why we need this? nn.Identity?

        return self.mlp_head(logits)


if __name__ == '__main__':
    
    v = NaViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=4096,
        dropout=0.,
        emb_dropout=0.,
        token_dropout_prob=0.1
    )

    imgs = [
        torch.randn(3, 256, 256),
        torch.randn(3, 224, 224),
        torch.randn(3, 192, 192),
        torch.randn(3, 64, 256),
        torch.randn(3, 256, 128),
    ]

    assert v(imgs).shape == (5, 1000)