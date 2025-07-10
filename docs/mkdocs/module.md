# Modules

## Atom
### Downsample
::: models.modules.downsample
    options:
        heading_level: 4

### Upsample
::: models.modules.upsample
    options:
        heading_level: 4

### Embedding
::: models.modules.embedding
    options:
        heading_level: 4

### Head
::: models.modules.head
    options:
        heading_level: 4

### Pre-process
::: models.modules.preprocessor
    options:
        heading_level: 4

### Post-process
::: models.modules.postprocessor
    options:
        heading_level: 4


### Fusion
::: models.modules.fusion
    options:
        heading_level: 4


## ResNet
::: models.modules.basic_resnet
    options:
        heading_level: 3
        show_root_toc_entry: false
        members:
            - BasicResidualBlock
            - ResidualBottleneck


## ViT

::: models.modules.basic_transformer
    options:
        heading_level: 3
        members:
            - SimpleVisionTransformerLayer
### Atom
::: models.modules.basic_transformer
    options:
        heading_level: 4
        show_submodules: true
        filters: ["!SimpleVisionTransformerLayer"]


## CBAM
::: models.modules.cbam
    options:
        heading_level: 3
        members:
            - BasicCAM
            - BasicSAM
            - BasicCBAM
            - CBAMBottleneck
### Atom
::: models.modules.basic_transformer
    options:
        heading_level: 4
        show_submodules: true
        filters: ["!BasicCAM","!BasicSAM","!BasicCBAM","!CBAMBottleneck",]


## Swin Transformer
::: models.modules.swin_transformer
    options:
        heading_level: 3
        members:
            - SwinTransformerLayer
            - SwinTransformerLayers
### Atom
::: models.modules.swin_transformer
    options:
        heading_level: 4
        show_submodules: true
        filters: ["!SwinTransformerLayer","!SwinTransformerLayers"]


## Channel Transformer
::: models.modules.channel_transformer
    options:
        heading_level: 3
        members:
            - ChannelTransformerLayer
            - ChannelTransformerLayers
            - UnetChannelTransformerLayers
### Atom
::: models.modules.channel_transformer
    options:
        heading_level: 4
        show_submodules: true
        filters: ["!ChannelTransformerLayer","!ChannelTransformerLayers","!UnetChannelTransformerLayers"]