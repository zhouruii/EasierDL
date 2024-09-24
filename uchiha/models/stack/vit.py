from torch import nn

from .base import Stack
from uchiha.models.builder import build_basemodule, build_embedding, build_preprocessor, build_head, MODEL


@MODEL.register_module()
class SimpleViT(Stack):
    def __init__(self,
                 embedding=None,
                 basemodule=None,
                 head=None):
        super().__init__(stacks=[{'embedding': embedding},
                                 {'basemodule': basemodule},
                                 {'head': head}])
        self.embedding: nn.Module = self.stacks[0]
        self.basemodule: nn.Module = self.stacks[1]
        self.head: nn.Module = self.stacks[2]

    def forward_features(self, x):
        # embedding
        out = self.embedding(x)

        # Transformer
        out = self.basemodule(out)

        return out

    def forward(self, x):
        out = self.forward_features(x)
        out = self.head(out)
        return out
