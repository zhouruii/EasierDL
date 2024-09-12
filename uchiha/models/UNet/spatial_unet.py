import torch

from torch import nn

from ..builder import build_preprocessor, build_embedding, build_basemodule, build_downsample, build_upsample, \
    build_bottleneck, build_head, build_fusion, MODEL
from ...utils.misc import strings_to_list


@MODEL.register_module()
class SpatialUNet(nn.Module):
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
        self.preprocessor = build_preprocessor(preprocessor)
        self.embedding = build_embedding(embedding)

        self.basemodule = build_basemodule(strings_to_list(basemodule))
        if hasattr(self.basemodule, 'layers'):
            self.layers = self.basemodule.layers
        else:
            self.layers = self.basemodule
        self.num_layers = len(self.layers) // 2
        self.encoder = self.layers[:self.num_layers]
        self.decoder = self.layers[self.num_layers:]

        self.downsample = build_downsample(downsample)
        self.upsample = build_upsample(upsample)
        self.bottleneck = build_bottleneck(bottleneck)

        self.fusion = build_fusion(strings_to_list(fusion))
        if hasattr(self.fusion, 'fusions'):
            self.fusion = self.fusion.fusions

        self.head = build_head(head)

    def forward_features(self, x):
        # preprocess
        if self.preprocessor:
            x = self.preprocessor(x)

        # embedding
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
        if self.bottleneck:
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
