# Dit, based on the facebook research

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
import math
from timm.layers.attention import Attention
from timm.layers.mlp import Mlp
# we use what we implement by ourselves

from simple_patchEmbed import PatchEmbed

# for attention, we use timm implementation
# because it's more efficient
# support qk_norm, fused_attention, qk_norm applied at the head level
# support Rope attention, rope applied also at the head level

# the mlp in timm, fc1 -> ac1 -> drop1 -> norm1 -> fc2 -> drop2

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

def exist(val):
    return val is not None

def default(val, d):
    return val if exist(val) else d


def modulate(x: Tensor, shift: Tensor, scale: Tensor):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# Embedding layer for conditioning, like time and class

class TimestepEmbedder(Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.frequency_embedding_size = frequency_embedding_size

    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        '''create the sinusoidal timestep embeddings

        :param float | int  t: a 1-d tensor of N indices, one per batch element, they can be fractional.
        :param  dim: the dim of the output
        :param int max_period: controls the the Maximum frequency of the embeddings, defaults to 10000
        '''        
        half_dim = dim // 2
        freq = torch.exp(
            -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float64) / half_dim
        )
        

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)

        return t_emb
    

class LabelEmbedder(Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding= dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_class = num_classes
        self.dropout_prob = dropout_prob
    
    def token_drop(self, labels, force_drop_ids = None):
        '''Dropout for token embeddings

        :param Tensor labels: The input labels
        :param Optional[Tensor] force_drop_ids: Specific token IDs to drop, defaults to None
        '''
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        
        else:
            drop_ids = force_drop_ids == 1
        
        labels = torch.where(drop_ids, self.num_class, labels)
        
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        '''Forward pass for label embedding

        :param Tensor labels: The input labels
        :param bool train: Whether the model is in training mode
        :param Optional[Tensor] force_drop_ids: Specific token IDs to drop, defaults to None
        '''

        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None): # one for training and one for inference
            labels = self.token_drop(labels, force_drop_ids)

        embeddings = self.embedding_table(labels)

        return embeddings
    


class DiTBlock(Module):
    '''
    A DiT block with ada layer norm (zero init) conditioning
    '''
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        '''
        Difference with what we saw in Lucidrain implementation about the attention:
        In lu's implementation, you proj you embedding vector to head * head_dim, you do multi-head attention, you use mlp to proj the lower dimension (usually smaller)

        In here, you simplify the two mlp into one. you can drop the mlps (two) used in the attention module
        we don't need to proj into high-dim, then proj it into low-dim.

        we just do multi-head attention, on the embedding vector, then use mlp (maybe sequential) to extract the info
        '''
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True,**block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate='tanh')

        # static analysis error: Cannot determine type of 'approx_gelu'
        self.mlp = Mlp(hidden_size, mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # this adaLN modulation layer, 6 times hidden_size. Wow, really Big mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6, bias=True)
        )

    def forward(self, x, c):
        '''Forward pass for DiTBlock

        :param Tensor x: Input tensor, x.shape should be (B, N, hidden_size)
        :param Tensor c: Conditioning tensor, c.shape should be (B, hidden_size) -> proj, and then chunk at the final dim, still hidden_size
        '''

        # so we need to unsqueeze, the drawback is that, we take every token in one sequence as the same (condition), scale, move, communication, gated, extract
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim = -1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x


class FinalLayer(Module):
    """
    The final layer for DiT
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        