# from now on, we only implement transformer once a day

import torch
from torch.nn import Module, ModuleList
from torch import Tensor, nn

import torch.nn.functional as F
from functools import partial
from einops import repeat
from einops.layers.torch import Rearrange

from nano_vit import Transformer, SimpleViT, posemb_sincos_2d




def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def divisible_by(numer, denom):
    return (numer % denom) == 0


class MAE(Module):
    def __init__(
        self,
        *,
        encoder: SimpleViT,
        decoder_dim,
        mask_ratio=0.75,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64,
    ):
        super().__init__()

        assert mask_ratio > 0 and mask_ratio < 1, "Invalid mask ratio"

        self.mask_ratio = mask_ratio

        # extract encoder params

        self.encoder = encoder

        patch_size = self.encoder.img_embedder.patch_size

        num_patches, encoder_dim = (
            self.encoder.img_embedder.num_patches,
            self.encoder.img_embedder.patch_to_embed[-2].out_features,
        )

        self.to_patch = self.encoder.img_embedder.to_patches
        self.patch_to_embed = self.encoder.img_embedder.patch_to_embed

        # weird name
        pixel_values_dim = self.encoder.img_embedder.patch_to_embed[-2].in_features 

        # decoder params
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)  # type: ignore

        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            decoder_depth, decoder_dim, decoder_heads, decoder_dim_head, decoder_dim * 4
        )

        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)

        self.to_pixel_values = nn.Linear(decoder_dim, pixel_values_dim) # type: ignore

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, img):
        
        patches = self.to_patch(img)
        dim_patch_height, dim_patch_width, _ = patches.shape[-3:]
    
        batch, num_patches = patches.shape[0], dim_patch_height * dim_patch_width


        tokens = self.patch_to_embed(patches)

        if self.encoder.pool == "cls":
            # the positional embedding for class token is not learned
            raise NotImplementedError("cls pooling not implemented for MAE")
        elif self.encoder.pool == "mean":
            tokens = tokens + posemb_sincos_2d(dim_patch_height, dim_patch_width, tokens.shape[-1], device=self.device)

        num_masked = int(self.mask_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=self.device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # encode only unmask token

        arange = partial(torch.arange, device=self.device)

        batch_range = arange(batch)[:, None] # Q: why need this [:, None]?
        unmasked_tokens = tokens[batch_range, unmasked_indices] # 

        # get the target patches

        masked_patches = patches[batch_range, masked_indices]

        
        encoded_tokens = self.encoder.transformer(unmasked_tokens)

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder pe to unmask tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask token
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=self.device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        mask_tokens  = decoded_tokens[batch_range, masked_indices]

        pred_pixel_values = self.to_pixel_values(mask_tokens)
        
        return F.mse_loss(pred_pixel_values, masked_patches)