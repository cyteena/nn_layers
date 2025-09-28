# based on the cross_vit from lucidrains, modified to fit my code style

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
# from torch.nested import nested_tensor

from einops import rearrange
from einops.layers.torch import Rearrange


from functools import partial
from typing import List


def exist(val):
    return val is not None


def default(val, d):
    return val if exist(val) else d


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def divisible_by(numer, denom):
    return (numer % denom) == 0


def FeedForward(dim, hidden_dim, dropout=0.0):
    return nn.Sequential(
        nn.LayerNorm(dim, bias=False),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, bias=False)

        dim_inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_k = nn.Linear(dim, dim_inner, bias=False)
        self.to_v = nn.Linear(dim, dim_inner, bias=False)

        self.q_norm = nn.LayerNorm(dim_head, bias=False)
        self.k_norm = nn.LayerNorm(dim_head, bias=False)

        self.to_out = nn.Linear(dim_inner, dim, bias=False)

        self.dropout = dropout

    def forward(self, x, context: Tensor | None = None):
        x = self.norm(x)  # pre-norm

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

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )

        out = transpose_head_seq(out).flatten(2)

        return self.to_out(out)


class TransformerBlock(Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0) -> None:
        super().__init__()
        self.attention = Attention(dim, heads, dim_head, dropout)
        self.feed_forward = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, context: Tensor | None = None):
        x = x + self.attention(x, context)
        x = x + self.feed_forward(x)
        return x
    
class Transformer(Module):
    def __init__(self, depth, dim, heads, dim_head, mlp_dim, dropout=0.) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, x, context: Tensor | None = None):
        for layer in self.layers:
            x = layer(x, context)
        return x


class ProjectionInOut(Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_proj = dim_in != dim_out

        self.project_in = nn.Linear(dim_in, dim_out) if need_proj else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_proj else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x


class CrossTransformerBlock(Module):
    def __init__(self, dim, heads, dim_head, sm_dim, lg_dim, dropout) -> None:
        super().__init__()
        self.sm_projection = ProjectionInOut(sm_dim, lg_dim, Attention(lg_dim, heads, dim_head, dropout))
        self.lg_projection = ProjectionInOut(lg_dim, sm_dim, Attention(sm_dim, heads, dim_head, dropout))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, _), (lg_cls, _) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        sm_cls = sm_cls + self.sm_projection(sm_cls, context=lg_tokens)
        lg_cls = lg_cls + self.lg_projection(lg_cls, context=sm_tokens)

        return (sm_cls, lg_cls)


class CrossTransformer(Module):
    def __init__(self, dim, head, dim_head, depth, sm_dim, lg_dim, dropout=0.0):
        super().__init__()
        self.layer = nn.ModuleList([
            CrossTransformerBlock(dim, head, dim_head, sm_dim, lg_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, sm_tokens, lg_tokens):
        for layer in self.layer:
            sm_tokens, lg_tokens = layer(sm_tokens, lg_tokens)
        return sm_tokens, lg_tokens


class MultiScaleEncoder(Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head=64,
        dropout=0.0,
    ) -> None:
        super().__init__()

        print(f"sm_enc_params: {sm_enc_params}")
        print(f"lg_enc_params: {lg_enc_params}")
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(
                ModuleList(
                    [
                        Transformer(dim = sm_dim, **sm_enc_params, dropout=dropout), # type: ignore
                        Transformer(dim = lg_dim, **lg_enc_params, dropout=dropout), # type: ignore
                        CrossTransformer(
                            sm_dim,
                            cross_attn_heads,
                            cross_attn_dim_head,
                            cross_attn_depth,
                            sm_dim,
                            lg_dim,
                            dropout,
                        ),
                    ]
                )
            )

    def forward(self, sm_tokens, lg_tokens):
        for module in self.layers:
            sm_enc, lg_enc, cross_attn = module # type: ignore
            sm_tokens = sm_enc(sm_tokens)
            lg_tokens = lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attn(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens


class ImageEmbedder(Module):
    def __init__(self, *, dim, image_size, patch_size, dropout=0.0, channels=3) -> None:
        super().__init__()
        self.channels = channels

        image_height, image_width = pair(image_size)

        assert divisible_by(image_height, patch_size) and divisible_by(
            image_width, patch_size
        ), "Image dimensions must be divisible by the patch size."

        self.patch_size = patch_size

        patch_height_dim, patch_width_dim = (
            (image_height // patch_size),
            (image_width // patch_size),
        )

        patch_dim = channels * (patch_size**2)

        self.to_patch = Rearrange(
            "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size
        )

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim, bias=False),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim, bias=False),
        )

        self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))

        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, imgs: List[Tensor]):
        _, device = len(imgs), self.device

        arange = partial(torch.arange, device=device)

        assert all([img.ndim == 3 and img.shape[0] == self.channels for img in imgs])

        all_patches = [self.to_patch(img) for img in imgs]

        positions = []

        for patches in all_patches:
            patch_height, patch_width = patches.shape[:2]

            # row and column indices
            # Q: why stack at dim -1? can we stack at dim = 0?
            hw_indices = torch.stack(
                torch.meshgrid(
                    arange(patch_height), arange(patch_width), indexing="ij"
                ),
                dim=-1,
            )

            # keep it same the "flatten" patches
            hw_indices = rearrange(hw_indices, "h w c -> (h w) c")

            positions.append(hw_indices)

        all_tokens = [rearrange(patches, "h w d -> (h w) d") for patches in all_patches]
        seq_lens = torch.tensor([i.shape[0] for i in all_tokens], device=device)
        if self.training and self.dropout.p > 0:
            keep_seq_lens = ((1 - self.dropout.p) * seq_lens).int().clamp(min=1)

            kept_tokens = []
            kept_positions = []

            for one_img_tokens, one_img_positions, seq_len, num_keep in zip(
                all_tokens, positions, seq_lens, keep_seq_lens
            ):
                # Q : why int() here? and what's the exactly meaning of topk?
                keep_indices = (
                    torch.randn(int(seq_len.item()), device=device)
                    .topk(int(num_keep.item()), dim=-1)
                    .indices
                )

                one_img_kept_tokens = one_img_tokens[keep_indices]
                one_img_kept_positions = one_img_positions[keep_indices]

                kept_tokens.append(one_img_kept_tokens)
                kept_positions.append(one_img_kept_positions)

            all_tokens, positions, seq_lens = kept_tokens, kept_positions, keep_seq_lens

        height_indices, width_indices = torch.cat(positions).unbind(
            dim=-1
        )  # (..., 2) -> (...), (...)
        # pick the height and width positional embedding
        height_embed, width_embed = (
            self.pos_embed_height[height_indices],
            self.pos_embed_width[width_indices],
        )

        pos_embed = (
            height_embed + width_embed
        )  # this can add together? we have the same shape?

        tokens = torch.cat(all_tokens, dim=0)

        tokens = self.to_patch_embedding(tokens)

        tokens = tokens + pos_embed

        # Q: what's the difference between this(dropout embedding) and dropout token?
        tokens = self.dropout(tokens) if self.training else tokens

        # nested_tensor can not be sliced, so we need to give up it
        max_seq_length = seq_lens.max().item()

        padding_tokens_list = []
        for token in tokens.split(seq_lens.tolist()):
            token = torch.cat((token, torch.zeros((max_seq_length - token.shape[0], token.shape[1]), device=device)))
            padding_tokens_list.append(token)

        tokens = torch.stack(padding_tokens_list, dim=0)
        assert tokens.ndim == 3

        return tokens


class CrossViT(Module):
    # we can use a config to manage all these parameters
    def __init__(
        self,
        *,
        img_size,
        num_class,
        sm_dim,
        lg_dim,
        sm_patch_size=12,
        sm_enc_depth=1,
        sm_enc_heads=8,
        sm_dim_head=64,
        sm_mlp_dim=2048,
        lg_patch_size=16,
        lg_enc_depth=4,
        lg_enc_heads=8,
        lg_enc_dim_head=64,
        lg_enc_mlp_dim=2048,
        cross_attn_depth=2,
        cross_attn_heads=8,
        cross_attn_dim_head=64,
        depth=3,
        dropout=0.1,
        emd_dropout=0.1,
        channels=3,
    ) -> None:
        super().__init__()
        self.sm_img_embedder = ImageEmbedder(
            dim=sm_dim,
            image_size=img_size,
            patch_size=sm_patch_size,
            dropout=dropout,
            channels=3,
        )
        self.lg_img_embedder = ImageEmbedder(
            dim=lg_dim,
            image_size=img_size,
            patch_size=lg_patch_size,
            dropout=dropout,
            channels=3,
        )

        self.multi_scale_encoder = MultiScaleEncoder(
            depth=depth,
            sm_dim=sm_dim,
            lg_dim=lg_dim,
            sm_enc_params=dict(
                depth=sm_enc_depth,
                heads=sm_enc_heads,
                dim_head=sm_dim_head,
                mlp_dim=sm_mlp_dim,
            ),
            lg_enc_params=dict(
                depth=lg_enc_depth,
                heads=lg_enc_heads,
                dim_head=lg_enc_dim_head,
                mlp_dim=lg_enc_mlp_dim,
            ),
            cross_attn_depth=cross_attn_depth,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dim_head=cross_attn_dim_head,
            dropout=dropout,
        )

        self.sm_mlp_head = nn.Sequential(
            nn.LayerNorm(sm_dim, bias=False), nn.Linear(sm_dim, num_class, bias=False)
        )
        self.lg_mlp_head = nn.Sequential(
            nn.LayerNorm(lg_dim, bias=False), nn.Linear(lg_dim, num_class, bias=False)
        )

    def forward(self, img):
        sm_tokens = self.sm_img_embedder(img)
        lg_tokens = self.lg_img_embedder(img)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)

        return sm_logits + lg_logits

if __name__ == "__main__":
    model = CrossViT(
        img_size=224,
        num_class=1000,
        sm_dim=192,
        lg_dim=384,
        sm_patch_size=16,
        sm_enc_depth=1,
        sm_enc_heads=3,
        sm_dim_head=64,
        sm_mlp_dim=768,
        lg_patch_size=32,
        lg_enc_depth=4,
        lg_enc_heads=6,
        lg_enc_dim_head=64,
        lg_enc_mlp_dim=1536,
        cross_attn_depth=2,
        cross_attn_heads=8,
        cross_attn_dim_head=64,
        depth=3,
        dropout=0.1,
        emd_dropout=0.1,
        channels=3,
    )

    imgs = [torch.randn(3, 224, 224), torch.randn(3, 128, 224)]
    logits = model(imgs)
    assert logits.shape == (2, 1000)
    print(logits.shape)  # should be [2, 1000]