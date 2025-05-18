import torch
from torch import nn

from ..builder import build_module, MODEL
from ...utils.misc import strings_to_list


@MODEL.register_module()
class ChannelUNet(nn.Module):
    # TODO 结构优化
    def __init__(self,
                 preprocessor=None,
                 embedding=None,
                 basemodule=None,
                 downsample=None,
                 bottleneck=None,
                 upsample=None,
                 fusion=None,
                 head=None, ):
        super().__init__()
        self.preprocessor = build_module(preprocessor)
        self.embedding = build_module(embedding)

        self.basemodule = build_module(strings_to_list(basemodule))
        if hasattr(self.basemodule, 'layers'):
            self.layers = self.basemodule.layers
        else:
            self.layers = self.basemodule
        self.num_layers = len(self.layers) // 2
        self.encoder = self.layers[:self.num_layers]
        self.decoder = self.layers[self.num_layers:]

        self.downsample = build_module(downsample)
        self.upsample = build_module(upsample)
        self.bottleneck = build_module(bottleneck)

        self.fusion = build_module(strings_to_list(fusion))
        if hasattr(self.fusion, 'fusions'):
            self.fusion = self.fusion.fusions

        self.head = build_module(head)

    def forward_features(self, x):
        if self.preprocessor:
            x = self.preprocessor(x)

        x = self.embedding(x)

        # encoder
        encoder = [x]
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                x = self.downsample(self.encoder[i](x))
                encoder.append(x)
            else:
                x = self.encoder[i](x)

        # bottleneck
        x = self.bottleneck(x)

        # decoder
        for i in range(self.num_layers):
            if i == 0:
                x = self.decoder[i](x)
                # fusion
                x = torch.cat([x, encoder[self.num_layers - 1 - i]], dim=-1)
                x = self.fusion[i](x)
            else:
                x = self.decoder[i](self.upsample(x))
                # fusion
                x = torch.cat([x, encoder[self.num_layers - 1 - i]], dim=-1)
                x = self.fusion[i](x)

        return x

    def forward(self, x):
        feats = self.forward_features(x)
        out = self.head(feats)
        return out
