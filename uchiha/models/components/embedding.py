from timm.layers import to_2tuple
from torch import nn

from ..builder import EMBEDDING


@EMBEDDING.register_module()
class PatchEmbedding(nn.Module):

    def __init__(self, img_size=4, patch_size=1, in_channel=330, embed_dim=512, norm_layer=None):
        super().__init__()
        assert img_size % patch_size == 0, \
            f'img_size:{img_size} cannot be divided by patch_size:{patch_size}'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_channel = in_channel
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(nn.Conv2d(in_channel, embed_dim, 1, 1, 0, bias=False),
                                  nn.GELU(),
                                  nn.Conv2d(embed_dim, embed_dim, 1, 1, 0, bias=False),
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
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


@EMBEDDING.register_module()
class TokenEmbedding(nn.Module):
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
