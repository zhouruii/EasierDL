from torch import nn

from ..builder import HEAD


@HEAD.register_module()
class FCHead(nn.Module):
    def __init__(self, embed_dim, pred_num):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, pred_num)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        # B,L,C = x.shape
        pooling = self.pooling(x.transpose(1, 2)).squeeze(2)
        out = self.head(pooling)
        return self.activate(out)
