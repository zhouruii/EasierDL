from torch import nn

from uchiha.models.builder import build_basemodule, build_embedding, build_preprocessor, build_head, MODEL


@MODEL.register_module()
class ViT1d(nn.Module):
    # TODO 结构优化
    def __init__(self,
                 preprocessor=None,
                 embedding=None,
                 basemodule=None,
                 head=None):
        super().__init__()
        self.preprocessor = build_preprocessor(preprocessor)

        self.embedding = build_embedding(embedding)

        self.transformer = build_basemodule(basemodule)

        self.mlp_head = build_head(head)

    def forward_features(self, x):
        # preprocess
        if self.preprocessor:
            x = self.preprocessor(x)

        # embedding
        x = self.embedding(x)

        # Transformer
        x = self.transformer(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        out = self.mlp_head(x)
        return out
