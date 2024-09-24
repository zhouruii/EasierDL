from timm.layers import to_2tuple
from torch import nn

from ..builder import EMBEDDING
from ...utils import build_norm


@EMBEDDING.register_module()
class PatchEmbedding(nn.Module):
    r""" Image to Patch Embedding

    (B, C, H, W) --> (B, L, _C)
    L = (H * W) / (patch_size ** 2)
    _C = embed_dim

    Args:
        img_size (int): Image size.  Default: 4.
        patch_size (int): Patch token size. Default: 1.
        in_channel (int): Number of input image channels. Default: 330.
        embed_dim (int): Number of Conv projection output channels. Default: 512.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=4, patch_size=1, in_channel=330, embed_dim=512, norm_layer=None):
        super().__init__()
        assert img_size % patch_size == 0, \
            f'img_size:{img_size} cannot be divided by patch_size:{patch_size}'

        self.img_size: tuple = to_2tuple(img_size)
        self.patch_size: tuple = to_2tuple(patch_size)
        patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_channel = in_channel
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        )

        norm_layer = build_norm(norm_layer)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


@EMBEDDING.register_module()
class TokenEmbedding(nn.Module):
    r""" Sequence to Patch Embedding

    (B, L, W) --> (B, L, _C)
    _C = embed_dim

    Args:
        in_channel (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """
    def __init__(self, in_channel, embed_dim, norm_layer):
        super().__init__()
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(nn.Linear(in_channel, embed_dim, bias=False),
                                  nn.GELU(),
                                  nn.Linear(embed_dim, embed_dim, bias=False),
                                  nn.GELU()
                                  )
        if norm_layer is not None:
            if norm_layer == 'nn.LayerNorm':
                self.norm = nn.LayerNorm(embed_dim)
            else:
                self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.flatten(2).transpose(1, 2)  # B C H W -> B L C

        x = self.proj(x)

        if self.norm is not None:
            x = self.norm(x)
        return x
