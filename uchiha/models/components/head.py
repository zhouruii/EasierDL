from torch import nn

from ..builder import HEAD


@HEAD.register_module()
class FCHead(nn.Module):
    def __init__(self,
                 embed_dim,
                 pred_num,
                 mode='sequence',
                 post_process=None):
        super().__init__()
        if mode == 'sequence':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(embed_dim, pred_num)

        if post_process == 'RELU':
            self.activate = nn.ReLU()
        else:
            self.activate = nn.Identity()

    def forward(self, x):
        if len(x.shape) == 3:
            # B,L,C = x.shape
            pooling = self.pooling(x.transpose(1, 2)).squeeze(2)
        else:
            # B,C,H,W = x.shape
            pooling = self.pooling(x).squeeze(-1).squeeze(-1)

        out = self.head(pooling)

        return self.activate(out)
