# 配置文件讲解

## 数据
关于数据加载相关的配置示例如下：
```yaml
data:
  train:
    dataset:
      type: HDF5MultiLevelRainHSIDataset
      h5_path: /home/disk2/ZR/datasets/AVIRIS/128/dataset.h5
      split_file: /home/disk2/ZR/datasets/AVIRIS/128/npy/train.txt
      pipelines:
        - type: RandomFlip
          directions: [original, horizontal, vertical, diagonal]
          probs: [0.3, 0.3, 0.3, 0.1]
        - type: RandomRotation
          angles: [0, 90, 180, 270]
          prob: 0.5
        - type: EasyToTensor
          mode: CHW
    dataloader:
      batch_size: 8
      shuffle: true
      num_workers: 8
      pin_memory: true
      persistent_workers: false
      prefetch_factor: 8
      drop_last: true
  val:
    dataset:
      type: HDF5MultiLevelRainHSIDataset
      h5_path: /home/disk2/ZR/datasets/AVIRIS/128/dataset.h5
      split_file: /home/disk2/ZR/datasets/AVIRIS/128/npy/val.txt
      pipelines:
        - type: EasyToTensor
          mode: CHW
    dataloader:
      batch_size: 8
      shuffle: false
```
其中，分别定义了训练和验证的数据加载，每个数据集都会有各自的参数，例如（`h5_path，split_file`）

每个数据集还会有对应的数据增强流程`pipelines`，例如上面的示例就采用了随机翻转，随机旋转等数据增强。

## 模型
模板目录中存放着预设的模型以及各类模块的使用示例

例如：
Swin Transformer使用：
```yaml
model:
  type: SwinTransformer
  embedding:
    type: PatchEmbedding
    img_size: 4
    patch_size: 2
    in_channel: 330
    embed_dim: 512
    norm_layer: nn.LayerNorm
  ape: false
  basemodule:
    type: SwinTransformerLayers
    dims: 512, 1024
    input_resolutions: 2, 1
    depths: 1, 1
    num_heads: 8, 16
    window_sizes: 1,1
    mlp_ratio: 4.0
    qkv_bias: true
    qk_scale: null
    drop_rate: 0.2
    attn_drop: 0.0
    drop_path_rate: 0.2
    norm_layer: nn.LayerNorm
    downsamples:
      type: PatchMerging
      input_resolution: 2
      in_channel: 512

  head:
    type: FCHead
    embed_dim: 1024
    pred_num: 1
```
只需在配置文件中进行相关参数改动，即可映射到实际使用的模型。

不仅如此，还可以多个模块进行组合使用，例如我们自定义的模型：
```yaml
model:
  type: Parallel
  preprocessor:
    type: DWT1d
    scales: 1
    wave: haar
    padding: zero
  parallels:
    - type: ChannelTransformer
      embedding:
        type: PatchEmbedding
        img_size: 4
        patch_size: 2
        in_channel: 165
        embed_dim: 256
        norm_layer: nn.LayerNorm
      basemodule:
        type: ChannelTransformerLayer
        dim: 256
        input_resolution: 2
        depth: 2
        num_heads: 16
        mlp_ratio: 2.0
        qkv_bias: true
        qk_scale: null
        drop_rate: 0.2 # drop_rate
        attn_drop: 0.0
        drop_path: 0.1
      head:
        type: FCHead
        embed_dim: 256
        pred_num: 1
    - type: ChannelTransformer
      embedding:
        type: PatchEmbedding
        img_size: 4
        patch_size: 2
        in_channel: 165
        embed_dim: 256
        norm_layer: nn.LayerNorm
      basemodule:
        type: ChannelTransformerLayer
        dim: 256
        input_resolution: 2
        depth: 1
        num_heads: 16
        mlp_ratio: 2.0
        qkv_bias: true
        qk_scale: null
        drop_rate: 0.2 # drop_rate
        attn_drop: 0.0
        drop_path: 0.1
      head:
        type: FCHead
        embed_dim: 256
        pred_num: 1
  postprocessor:
    type: WeightedSum
    weights:
      - 0.8
      - 0.2
```
该配置文件，就是用了并行的结构，并行的每一条支路都是一个`Channel Transformer`的串行结构。此外，骨干网络前后都会放置一个处理器，该模块仍然可以自定义！

## 训练参数

训练时需要有众多的超参数需要调整，使用示例如下：
```yaml
train:
  total_epoch: &total_epoch 100
  use_grad_clip: true
  print_freq: 50

  loss:
    type: MSELoss

  optimizer:
    type: AdamW
    lr: 3.0e-4
    weight_decay: 1.0e-4
    betas: [ 0.9, 0.999 ]


  scheduler:
    type: LinearWarmupCosineLR
    total_epochs: *total_epoch
    warmup_epochs: 10
    warmup_start_lr: 3.0e-4
    min_lr: 1.0e-6

val:
  val_freq: 1
  metric: psnr

checkpoint:
  save_freq: 10
  resume_from: null
  auto_resume: false
```
上述配置指定了众多超参设置，例如，初始学习率，学习率调整，优化器，迭代数等等。