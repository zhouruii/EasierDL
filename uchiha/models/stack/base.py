from torch import nn

from uchiha.models.builder import (build_basemodule, build_embedding,
                                   build_preprocessor, build_head, MODEL)


@MODEL.register_module()
class Stack(nn.Module):
    def __init__(self,
                 preprocessor=None,
                 embedding=None,
                 stacks=None,
                 head=None):
        super().__init__()
        self.preprocessor = build_preprocessor(preprocessor)

        self.embedding = build_embedding(embedding)

        self.basemodules = nn.ModuleList()
        for module in stacks:
            self.basemodules.append(build_basemodule(module))

        self.head = build_head(head)

    def forward_features(self, x):
        # preprocess
        if self.preprocessor:
            x = self.preprocessor(x)

        # embedding
        if self.embedding:
            x = self.embedding(x)

        # Transformer
        for module in self.basemodules:
            x = module(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        out = self.head(x)
        return out
