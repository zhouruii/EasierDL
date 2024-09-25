# Modules

## Atom
### Downsample
::: models.components.downsample
    options:
        heading_level: 4

### Upsample
::: models.components.upsample
    options:
        heading_level: 4

### Embedding
::: models.components.embedding
    options:
        heading_level: 4

### Head
::: models.components.head
    options:
        heading_level: 4

### Pre-process
::: models.components.preprocessor
    options:
        heading_level: 4

### Post-process
::: models.components.postprocessor
    options:
        heading_level: 4

### Bottleneck(U-Net)
::: models.components.bottleneck
    options:
        heading_level: 4

### Fusion
::: models.components.fusion
    options:
        heading_level: 4


## ResNet
::: models.basemodules.basic_resnet
    options:
        heading_level: 3
        show_root_toc_entry: false
        members:
            - BasicResidualBlock
            - ResidualBottleneck


## ViT

::: models.basemodules.basic_transformer
    options:
        heading_level: 3
        members:
            - SimpleVisionTransformerLayer
### Atom
::: models.basemodules.basic_transformer
    options:
        heading_level: 4
        show_submodules: true
        filters: ["!SimpleVisionTransformerLayer"]


## CBAM
::: models.basemodules.cbam
    options:
        heading_level: 3
        members:
            - BasicCAM
            - BasicSAM
            - BasicCBAM
            - CBAMBottleneck
### Atom
::: models.basemodules.basic_transformer
    options:
        heading_level: 4
        show_submodules: true
        filters: ["!BasicCAM","!BasicSAM","!BasicCBAM","!CBAMBottleneck",]


## Swin Transformer
::: models.basemodules.swin_transformer
    options:
        heading_level: 3
        members:
            - SwinTransformerLayer
            - SwinTransformerLayers
### Atom
::: models.basemodules.swin_transformer
    options:
        heading_level: 4
        show_submodules: true
        filters: ["!SwinTransformerLayer","!SwinTransformerLayers"]


## Channel Transformer
::: models.basemodules.channel_transformer
    options:
        heading_level: 3
        members:
            - ChannelTransformerLayer
            - ChannelTransformerLayers
            - UnetChannelTransformerLayers
### Atom
::: models.basemodules.channel_transformer
    options:
        heading_level: 4
        show_submodules: true
        filters: ["!ChannelTransformerLayer","!ChannelTransformerLayers","!UnetChannelTransformerLayers"]