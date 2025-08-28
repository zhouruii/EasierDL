import torch
from torch import nn
from torch.nn import init

from ..builder import MODEL, build_module


def fill_former_cfg(cfg, channels_former, channels_prior, num_heads, num_blocks, window_size):
    filled_cfg = cfg.copy()
    filled_cfg['in_channels'] = channels_former
    filled_cfg['num_heads'] = num_heads
    filled_cfg['num_blocks'] = num_blocks
    filled_cfg['window_size'] = window_size
    filled_cfg['prior_cfg']['in_channels'] = channels_prior
    filled_cfg['ffn_cfg']['in_channels'] = channels_former

    return filled_cfg


def fill_sampling_cfg(cfg, sampling_type, in_channels):
    filled_cfg = cfg.copy()
    filled_cfg['type'] = sampling_type
    filled_cfg['in_channels'] = in_channels

    return filled_cfg


def create_sampling_module(former_samplings, prior_samplings, sampling_cfg, sampling_type, channels_former,
                           channels_prior):
    former_samplings.append(build_module(fill_sampling_cfg(cfg=sampling_cfg,
                                                           sampling_type=sampling_type,
                                                           in_channels=channels_former)))
    prior_samplings.append(build_module(fill_sampling_cfg(cfg=sampling_cfg,
                                                          sampling_type=sampling_type,
                                                          in_channels=channels_prior)))


# last decoder differs from v1
@MODEL.register_module()
class HDRFormerV2(nn.Module):
    """
    Hyperspectral DeRaining Transformer model.
    """

    def __init__(self,
                 in_channels=224,
                 out_channels=224,
                 prior_extractor=None,
                 prev_band_selector=None,
                 post_band_selector=None,
                 embedding_cfg=None,
                 fusion_cfg=None,
                 sampling_cfg=None,
                 transformer_cfg=None,
                 num_refinement_blocks=None,
                 reconstruction=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prior_extractor = build_module(prior_extractor)
        self.embedding = build_module(embedding_cfg)
        self.prev_band_selector = build_module(prev_band_selector)
        self.post_band_selector = build_module(post_band_selector)
        self.reconstruction = build_module(reconstruction)

        self.prior_dim = len(prior_extractor.get('split_ratios')) * 2
        self.embed_dim = embedding_cfg.get('embed_dim')
        self.num_heads = transformer_cfg.get('num_heads')
        self.window_sizes = transformer_cfg.get('window_size')
        self.num_blocks = transformer_cfg.get('num_blocks')
        self.num_refinement_blocks = num_refinement_blocks
        self.num_layers = len(self.num_blocks)

        downsample_type = sampling_cfg.pop('downsample')
        upsample_type = sampling_cfg.pop('upsample')

        cfg = transformer_cfg.copy()
        self.former_samplings = nn.ModuleList()
        self.prior_samplings = nn.ModuleList()

        # encoder
        self.encoders = nn.ModuleList()
        for i in range(0, self.num_layers - 1):
            channels_former = self.embed_dim * (2 ** i)
            channels_prior = self.prior_dim * (2 ** i)
            filled_former_cfg = fill_former_cfg(cfg=cfg,
                                                channels_former=channels_former,
                                                channels_prior=channels_prior,
                                                num_heads=self.num_heads[i],
                                                num_blocks=self.num_blocks[i],
                                                window_size=self.window_sizes[i])

            self.encoders.append(build_module(filled_former_cfg))
            create_sampling_module(self.former_samplings, self.prior_samplings, sampling_cfg, downsample_type,
                                   channels_former, channels_prior)

        # bottleneck
        self.bottleneck = build_module(fill_former_cfg(cfg=cfg,
                                                       channels_former=self.embed_dim * (2 ** (self.num_layers - 1)),
                                                       channels_prior=self.prior_dim * (2 ** (self.num_layers - 1)),
                                                       num_heads=self.num_heads[self.num_layers - 1],
                                                       num_blocks=self.num_blocks[self.num_layers - 1],
                                                       window_size=self.window_sizes[self.num_layers - 1]))

        # decoder
        self.decoders = nn.ModuleList()
        for i in range(self.num_layers, 2 * self.num_layers - 1):
            channels_former = self.embed_dim * (2 ** (self.num_layers - 2 - i % self.num_layers))
            channels_prior = self.prior_dim * (2 ** (self.num_layers - 2 - i % self.num_layers))
            filled_former_cfg = fill_former_cfg(cfg=cfg,
                                                channels_former=channels_former,
                                                channels_prior=channels_prior,
                                                num_heads=self.num_heads[i % self.num_layers],
                                                num_blocks=self.num_blocks[i % self.num_layers],
                                                window_size=self.window_sizes[i % self.num_layers])
            create_sampling_module(self.former_samplings, self.prior_samplings, sampling_cfg, upsample_type,
                                   channels_former * 2, channels_prior * 2)
            if i != 2 * self.num_layers - 2:
                self.decoders.append(build_module(filled_former_cfg))
            else:
                self.decoders.append(build_module(fill_former_cfg(cfg=cfg,
                                                                  channels_former=self.embed_dim * 2,
                                                                  channels_prior=self.prior_dim * 2,
                                                                  num_heads=self.num_heads[0],
                                                                  num_blocks=self.num_blocks[
                                                                                 0] + self.num_refinement_blocks,
                                                                  window_size=self.window_sizes[0])))

        self.former_fusions = nn.ModuleList()
        complete_fusion_cfg = fusion_cfg.copy()
        for i in range(self.num_layers - 1, 1, -1):
            complete_fusion_cfg['in_channels'] = self.embed_dim * (2 ** i)
            self.former_fusions.append(build_module(complete_fusion_cfg))

        self.prior_fusions = nn.ModuleList()
        complete_fusion_cfg = fusion_cfg.copy()
        for i in range(self.num_layers - 1, 1, -1):
            complete_fusion_cfg['in_channels'] = self.prior_dim * (2 ** i)
            self.prior_fusions.append(build_module(complete_fusion_cfg))

        self.out_proj = nn.Sequential(
            nn.Conv2d(self.embed_dim * 2, abs(self.out_channels + self.embed_dim * 2) // 2, 1, bias=False),
            nn.Conv2d(abs(self.out_channels + self.embed_dim * 2) // 2, self.out_channels, 1, bias=False)
        )

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=0.02)
            # init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    def forward(self, x):
        shortcut = x
        depth = len(self.encoders)
        rcp_prior = self.prior_extractor(x)
        feat = self.embedding(self.prev_band_selector(x))

        # encode
        to_fuse_feats = []
        to_fuse_rcp_priors = []
        for i in range(depth):
            encoder = self.encoders[i]
            former_downsample = self.former_samplings[i]
            prior_downsample = self.prior_samplings[i]
            to_fuse_feat, to_fuse_rcp_prior = encoder(feat, rcp_prior)
            feat = former_downsample(to_fuse_feat)
            rcp_prior = prior_downsample(to_fuse_rcp_prior)
            to_fuse_feats.append(to_fuse_feat)
            to_fuse_rcp_priors.append(to_fuse_rcp_prior)

        # bottleneck
        feat, rcp_prior = self.bottleneck(feat, rcp_prior)

        # decode
        for i in range(depth):
            decoder = self.decoders[i]
            former_upsample = self.former_samplings[i + depth]
            prior_upsample = self.prior_samplings[i + depth]

            feat = former_upsample(feat)
            rcp_prior = prior_upsample(rcp_prior)
            if i < depth - 1:
                former_fusion = self.former_fusions[i]
                prior_fusion = self.prior_fusions[i]
                feat = former_fusion(feat, to_fuse_feats[depth - 1 - i])
                rcp_prior = prior_fusion(rcp_prior, to_fuse_rcp_priors[depth - 1 - i])
            else:
                feat = torch.cat((feat, to_fuse_feats[0]), dim=1)
                rcp_prior = torch.cat((rcp_prior, to_fuse_rcp_priors[0]), dim=1)
            feat, rcp_prior = decoder(feat, rcp_prior)

        feat = self.out_proj(feat)
        feat = self.post_band_selector(feat)
        out = self.reconstruction(feat, shortcut)

        return out
