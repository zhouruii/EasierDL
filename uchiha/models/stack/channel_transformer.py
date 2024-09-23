from torch import nn

from ..builder import (build_preprocessor, build_embedding, build_basemodule,
                       build_downsample, build_head, MODEL)
from ...utils.misc import strings_to_list


@MODEL.register_module()
class ChannelTransformer(nn.Module):
    # TODO 结构优化
    def __init__(self,
                 preprocessor=None,
                 embedding=None,
                 basemodule=None,
                 downsample=None,
                 head=None, ):
        super().__init__()
        self.preprocessor = build_preprocessor(preprocessor)
        self.embedding = build_embedding(embedding)

        self.basemodule = build_basemodule(strings_to_list(basemodule))
        if hasattr(self.basemodule, 'layers'):
            self.layers = self.basemodule.layers
            self.num_layers = len(self.layers)
        else:
            self.layers = self.basemodule
            self.num_layers = 1

        self.downsample = build_downsample(downsample)

        self.head = build_head(head)

    def forward_features(self, x):
        # preprocess
        if self.preprocessor:
            x = self.preprocessor(x)

        # embedding
        x = self.embedding(x)

        # core
        if self.num_layers > 1:
            for i in range(self.num_layers):
                if i < self.num_layers - 1:
                    x = self.downsample(self.layers[i](x))
                else:
                    x = self.layers[i](x)
        else:
            if self.downsample:
                x = self.downsample(self.layers(x))
            else:
                x = self.layers(x)

        return x

    def forward(self, x):
        out = self.forward_features(x)
        if self.head:
            out = self.head(out)
        return out
