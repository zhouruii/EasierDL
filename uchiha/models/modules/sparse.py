import torch
from torch import nn


class SparsifyAttention(nn.Module):
    STRATEGIES = ['MultiscaleTopK', 'AdaptiveReLU2']

    def __init__(self, strategy='MultiscaleTopK'):
        super().__init__()
        assert strategy in self.STRATEGIES or strategy is None, f'strategy:{strategy} not supported yet !'

        if strategy == 'MultiscaleTopK':
            self.sparse_opt = MultiscaleTopKSparseAttention()
        elif strategy == 'AdaptiveReLU2':
            self.sparse_opt = AdaptiveReLU2SparseAttention()
        else:
            self.sparse_opt = nn.Identity()

    def forward(self, attn):
        # att.shape (B, nH, N, N) N = C or HW
        return self.sparse_opt(attn)


class MultiscaleTopKSparseAttention(nn.Module):
    """
    refer to: DRS-former (CVPR 2023)
    """

    def __init__(self):
        super(MultiscaleTopKSparseAttention, self).__init__()

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, attn):
        # att.shape (B, nH, C, C)
        b, nh, C, C = attn.shape
        mask1 = torch.zeros(b, nh, C, C, device=attn.device, requires_grad=False)
        mask2 = torch.zeros(b, nh, C, C, device=attn.device, requires_grad=False)
        mask3 = torch.zeros(b, nh, C, C, device=attn.device, requires_grad=False)
        mask4 = torch.zeros(b, nh, C, C, device=attn.device, requires_grad=False)

        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        final_attn = attn1 * self.attn1 + attn2 * self.attn2 + attn3 * self.attn3 + attn4 * self.attn4

        return final_attn


class AdaptiveReLU2SparseAttention(nn.Module):
    """
    refer to: AST (CVPR 2024)
    """

    def __init__(self):
        super(AdaptiveReLU2SparseAttention, self).__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, attn):
        # att.shape (B, nH, N, N) N = C or HW
        attn0 = self.softmax(attn)
        attn1 = self.relu(attn) ** 2

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        final_attn = attn0 * w1 + attn1 * w2

        return final_attn
