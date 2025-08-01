from typing import Union

from torch import nn

from ..builder import MODEL
from .base import Stack
from ...utils.misc import strings_to_list


@MODEL.register_module()
class ChannelTransformer(Stack):
    """ Channel Transformer Network

    Args:
        embedding (dict): Config information for building the embedding. Default: None.
        basemodule (dict): Config information for building the basemodule. Default: None.
        head (dict): Config information for building the head. Default: None.
    """

    def __init__(self,
                 embedding=None,
                 basemodule=None,
                 head=None):

        basemodule = strings_to_list(basemodule)
        super().__init__(stacks=[{'embedding': embedding},
                                 {'basemodule': basemodule},
                                 {'head': head}])
        self.embedding: nn.Module = self.stacks[0]
        self.layers: Union[nn.ModuleList, nn.Module] = self.stacks[1]
        self.head: nn.Module = self.stacks[2]

        if not isinstance(self.layers, nn.ModuleList):
            self.layers = nn.ModuleList([self.layers])

    def forward_features(self, x):
        # embedding
        out = self.embedding(x)

        # core
        for layer in self.layers:
            out = layer(out)

        return out

    def forward(self, x):
        out = self.forward_features(x)
        if self.head:
            out = self.head(out)
        return out
