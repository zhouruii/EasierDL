import torch
from timm.layers import trunc_normal_
from torch import nn

from ..builder import MODEL
from .base import Stack
from ..modules.common import build_norm
from ...utils.misc import strings_to_list


@MODEL.register_module()
class SwinTransformer(Stack):
    """ Swin-Transformer Network

    Args:
        embedding (dict): Config information for building the embedding. Default: None.
        basemodule (dict): Config information for building the basemodule. Default: None.
        ape (bool): Whether to use absolute position encoding. Default: False.
        head (dict): Config information for building the head. Default: None.
    """

    def __init__(self,
                 embedding=None,
                 ape=False,
                 basemodule=None,
                 head=None):

        basemodule = strings_to_list(basemodule)
        super().__init__(stacks=[{'embedding': embedding},
                                 {'basemodule': basemodule},
                                 {'head': head}])
        self.embedding: nn.Module = self.stacks[0]
        self.basemodule: nn.Module = self.stacks[1]
        self.head: nn.Module = self.stacks[2]

        # absolute position embedding
        self.ape = ape
        num_patches = self.embedding.num_patches
        embed_dim = self.embedding.embed_dim
        drop_rate = basemodule.get('drop_rate')
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if hasattr(self.basemodule, 'layers'):
            self.layers = self.basemodule.layers
            self.num_layers = len(self.layers)
        else:
            self.layers = nn.ModuleList([self.basemodule])
            self.num_layers = 1

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        norm_layer = build_norm(basemodule.get('norm_layer'))
        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

    def forward_features(self, x):
        # embedding
        out = self.embedding(x)

        # ape
        if self.ape:
            out = out + self.absolute_pos_embed
        out = self.pos_drop(out)

        # core
        for layer in self.layers:
            out = layer(out)

        # norm
        out = self.norm(out)

        return out

    def forward(self, x):
        out = self.forward_features(x)
        if self.head:
            out = self.head(out)
        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):
        patches_resolution = self.embedding.patches_resolution
        num_classes = self.head.pred_num
        flops = 0
        flops += self.embedding.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * patches_resolution[0] * patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * num_classes
        return flops
