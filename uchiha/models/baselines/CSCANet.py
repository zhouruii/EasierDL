import math

import torch.nn as nn
import torch
import numbers
from einops import rearrange

from ..builder import MODEL


## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        a = x - mu
        b = torch.sqrt(sigma + 1e-5)
        return a / b * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode='channel'):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.mode = mode
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q(x)  # image
        k = self.k(x)  # event
        v = self.v(x)  # event

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class CatSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode='channel'):
        super(CatSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.mode = mode
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q(x)  # image
        k = self.k(x)  # event
        v = self.v(x)  # event

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode='channel'):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.mode = mode
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q(x)
        k = self.k(y)
        v = self.v(y)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfTransformer(nn.Module):
    def __init__(self, dim, num_heads=6, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(SelfTransformer, self).__init__()
        # mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = SelfAttention(dim, num_heads, bias)
        # self.lnorm = nn.LayerNorm(dim)
        # self.ffn = Mlp(dim, mlp_hidden_dim)

    def forward(self, x):
        # b, c, h, w = x1.shape
        out = self.attn(self.norm1(x))  # b, c, h, w
        out = x + out
        # out = to_3d(out)  # b, h*w, c
        # out = out + self.ffn(self.lnorm(out))
        # out = to_4d(out, h, w)
        return out


class CatSelfTransformer(nn.Module):
    def __init__(self, dim, num_heads=6, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(CatSelfTransformer, self).__init__()
        # mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.attn = CatSelfAttention(dim, num_heads, bias)
        # self.lnorm = nn.LayerNorm(dim)
        # self.ffn = Mlp(dim, mlp_hidden_dim)

    def forward(self, x1, x2):
        assert x1.shape == x2.shape, 'the shape of image doesnt equal to event'
        # b, c, h, w = x1.shape
        out = self.attn(torch.cat([self.norm1(x1), self.norm2(x2)], dim=1))  # b, c, h, w
        out = x1 + out
        # out = to_3d(out)  # b, h*w, c
        # out = out + self.ffn(self.lnorm(out))
        # out = to_4d(out, h, w)
        return out


class CrossTransformer(nn.Module):
    def __init__(self, dim, num_heads=6, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(CrossTransformer, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention(dim, num_heads, bias)

    def forward(self, x1, x2):
        # x1 -> q
        # x2 -> k v
        out = self.attn(self.norm1(x1), self.norm2(x2))  # b, c, h, w
        out = x1 + out
        return out


class Basic_Residual_Block(nn.Module):
    def __init__(self, BCB_channels=306, BCB_feature=306):
        super(Basic_Residual_Block, self).__init__()
        self.BCB_channels = BCB_channels
        self.input_channel = BCB_feature
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=self.BCB_channels, kernel_size=3, padding=1),
            nn.ReLU())
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=self.BCB_channels, out_channels=self.BCB_channels, kernel_size=1), nn.ReLU())
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=self.BCB_channels, out_channels=self.BCB_channels, kernel_size=3, padding=1),
            nn.ReLU())
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=self.BCB_channels, kernel_size=1), nn.ReLU())

    def forward(self, BCB_feature):
        feature = self.conv_layer1(BCB_feature)
        feature = self.conv_layer2(feature)
        feature = self.conv_layer3(feature)
        identity_feature = self.conv_layer4(BCB_feature)
        output = torch.add(feature, identity_feature)
        return output


class Multiscale_Block(nn.Module):
    def __init__(self, input_channel=306):
        super(Multiscale_Block, self).__init__()
        self.input_channel = input_channel
        self.conv_sp_atten_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=306, kernel_size=3, padding=1), nn.Sigmoid())
        self.conv_sp_atten_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=306, kernel_size=5, padding=2), nn.Sigmoid())
        self.conv_atten = nn.Sequential(
            nn.Conv2d(in_channels=2 * self.input_channel, out_channels=306, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, Input_features):
        follow_feature1 = self.conv_sp_atten_1(Input_features)
        follow_feature2 = self.conv_sp_atten_2(Input_features)
        multiscale_feature = torch.cat((follow_feature1, follow_feature2), dim=1)
        multiscale_feature = self.conv_atten(multiscale_feature)
        return multiscale_feature


@MODEL.register_module()
class CSCANet(nn.Module):
    def __init__(self, input_image_channel=305):
        super(CSCANet, self).__init__()
        self.input_image_channel = input_image_channel
        self.first_branch_channel = 306 // 3
        self.input_transformer = math.floor(self.first_branch_channel / 6) * 6
        self.resinput_transformer = self.first_branch_channel - self.input_transformer

        self.attention_1 = CrossTransformer(self.input_transformer)
        self.attention_2 = CrossTransformer(self.input_transformer)
        self.attention_3 = CrossTransformer(self.input_transformer)
        self.attention_4 = CrossTransformer(self.input_transformer)
        self.attention_5 = CrossTransformer(self.input_transformer)
        self.attention_6 = CrossTransformer(self.input_transformer)
        self.attention_7 = CrossTransformer(self.input_transformer)
        self.attention_8 = CrossTransformer(self.input_transformer)
        self.attention_9 = CrossTransformer(self.input_transformer)
        self.attention_10 = CrossTransformer(self.input_transformer)

        self.Basic_Residual_1 = Basic_Residual_Block(BCB_feature=306)
        self.Multiscale_Block_1 = Multiscale_Block()
        self.Basic_Residual_2 = Basic_Residual_Block(BCB_feature=306)

        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels=204, out_channels=102, kernel_size=1), nn.ReLU())
        self.conv_layer2 = nn.Sequential(nn.Conv2d(in_channels=204, out_channels=102, kernel_size=1), nn.ReLU())
        self.conv_layer3 = nn.Sequential(nn.Conv2d(in_channels=102, out_channels=102, kernel_size=1), nn.ReLU())
        self.conv_layer4 = nn.Sequential(nn.Conv2d(in_channels=204, out_channels=102, kernel_size=1), nn.ReLU())
        self.conv_layer5 = nn.Sequential(nn.Conv2d(in_channels=204, out_channels=102, kernel_size=1), nn.ReLU())
        self.conv_layer6 = nn.Sequential(nn.Conv2d(in_channels=102, out_channels=102, kernel_size=1), nn.ReLU())

    def forward(self, Input_HSI):
        # _, band, _, _ = Input_HSI.shape     # 获得HSI的通道数

        first_branch_channel = self.first_branch_channel
        input_transformer = self.input_transformer

        first_branch_input = Input_HSI[:, 0:first_branch_channel, :, :]
        second_branch_input = Input_HSI[:, first_branch_channel:2 * first_branch_channel, :, :]
        third_branch_input = second_branch_input
        third_branch_input[:, 0:first_branch_channel - 1, :, :] = Input_HSI[:, 2 * first_branch_channel:, :, :]
        third_branch_input[:, first_branch_channel - 1, :, :] = third_branch_input[:, first_branch_channel - 2, :, :]

        first_branch_output = torch.cat((self.attention_1(first_branch_input, first_branch_input),
                                         self.attention_2(first_branch_input, second_branch_input),), dim=1)
        # print(first_branch_input.size())
        # print(first_branch_output.size())
        first_branch_output = self.conv_layer1(first_branch_output) + first_branch_input

        second_branch_output = torch.cat((self.attention_3(second_branch_input, second_branch_input),
                                          self.attention_4(second_branch_input, third_branch_input)), dim=1)
        second_branch_output = self.conv_layer2(second_branch_output) + second_branch_input

        third_branch_output = self.attention_5(third_branch_input, third_branch_input) + third_branch_input
        third_branch_output = self.conv_layer3(third_branch_output)

        first_branch_output1 = torch.cat((self.attention_6(first_branch_output, first_branch_output),
                                          self.attention_7(first_branch_output, second_branch_output),), dim=1)
        first_branch_output = self.conv_layer4(first_branch_output1) + first_branch_output

        second_branch_output1 = torch.cat((self.attention_8(second_branch_output, second_branch_output),
                                           self.attention_9(second_branch_output, third_branch_output)), dim=1)
        second_branch_output = self.conv_layer5(second_branch_output1) + second_branch_output

        third_branch_output = self.attention_10(third_branch_output, third_branch_output) + third_branch_output
        third_branch_output = self.conv_layer6(third_branch_output)

        merge_output = torch.cat((first_branch_output, second_branch_output, third_branch_output), dim=1)
        merge_1 = self.Basic_Residual_1(merge_output)
        merge_1 = self.Multiscale_Block_1(merge_1)
        merge_1 = self.Basic_Residual_2(merge_1)

        output = merge_1[:, 0:305, :, :]
        return output


if __name__ == '__main__':
    mcplb = CSCANet(input_image_channel=305).cuda()
