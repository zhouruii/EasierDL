# DIY 模型配置与使用指南

本文档展示如何通过配置文件自定义并配置数据、网络、训练等各组件，帮助你顺利地运行一个自定义深度学习项目。

本指南基于作者实验室的工作经验，可模拟应用到其他深度学习场景。

---

## 一、环境准备

* 确保按照项目文档完成环境搭建
* 根据项目要求确保 PyTorch 和 CUDA 版本匹配
* 别的依赖请实施 `requirements.txt`

可以预处理好数据或者仅添加些示例样本以便快速运行

---

## 二、数据配置

### 基本内容

* Dataset 的类型和路径
* 数据增强策略
* DataLoader 参数

### 示例 YAML 配置

```yaml
data:
  train:
    dataset:
      type: HDF5MultiLevelRainHSIDataset
      h5_path: /path/to/train/dataset.h5
      split_file: /path/to/train.txt
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
      h5_path: /path/to/val/dataset.h5
      split_file: /path/to/val.txt
      pipelines:
        - type: EasyToTensor
          mode: CHW
    dataloader:
      batch_size: 8
      shuffle: false
```

### 配置解读

以上配置包含了训练集与验证集的加载与数据增强等流程。通过指定data属性

工作流为定义一个Dataset类，配置文件中的名字即为类名，后续的配置是该类所需要的参数。例如
- h5_path就是高光谱文件的源路径
- pipelines则定义了数据增强的流程，数据增强的定义与Dataset类相似，以上配置则定义了对数据进行随机翻转，随机旋转以及最后的张量转换

定义好Dataset类后需要使用DataLoader进行数据加载，直接在配置文件中定义相关的参数即可，例如：
- batch size
- num workers

---

## 三、网络模型配置

### 基本示例

```yaml
model:
  type: ResNet
  block: BasicResidualBlock
  layers: [2, 2, 2, 2]
  in_channel: 3
  num_classes: 10
```
为了演示方便，上述示例是一个非常简单且经典的ResNet。原理与上述数据部分类似，通过指定model属性

从配置文件就可以直观的感受到网络模型的相关参数，例如：
- 输入数据的通道数为3（适用于RGB图片）
- 网络共有四层，每层中包含了2个残差块
- 最后需要将图像数据映射到10分类问题

### 复杂网络配置示例

```yaml
model:
  type: HDRFormer
  in_channels: *num_bands
  out_channels: *num_bands
  prior_extractor:
    type: GroupRCP
    split_bands: [50, 100, 150]
  prev_band_selector:
    type: BCAM
    in_channels: *num_bands
    ratio: 0.5
    min_proj: false
  post_band_selector:
    type: BCAM
    in_channels: *num_bands
    ratio: 0.5
    min_proj: false
  embedding_cfg:
    type: OverlapPatchEmbedding
    in_channels: *num_bands
    embed_dim: *embed_dim
    bias: false
  fusion_cfg:
    type: CatPWConvFusion
  sampling_cfg:
    downsample: PixelShuffleDownsample
    upsample: PixelShuffleUpsample
    factor: 2
  transformer_cfg:
    type: SelfCrossAttentionLayer
    num_heads: [2, 4, 4, 8]
    num_blocks: [4, 6, 6, 8]
    freq_cfg:
      type: DWT
      J: 1
      wave: haar
      mode: reflect
    prior_cfg:   # forward for prior
      type: GRCPBranch
      strategy: dilation
    sparse_strategy: MultiscaleTopK
    ffn_cfg:
      type: LeFF
      ratio: 4
      use_eca: true
    ln_bias: true
  reconstruction:
    type: FreqDecouplingReconstruction
    d: 1
    in_channels: *num_bands
    filter_kernel_size: 3
    filter_groups: 8
    norm: GN
```

---

## 四、训练配置

### 基础配置

```yaml
train:
  total_epoch: 100
  use_grad_clip: true
  print_freq: 50

  loss:
    type: MSELoss

  optimizer:
    type: AdamW
    lr: 3.0e-4
    weight_decay: 1.0e-4

  scheduler:
    type: LinearWarmupCosineLR
    total_epochs: 100
    warmup_epochs: 10
    warmup_start_lr: 3.0e-4
    min_lr: 1.0e-6
```

### 配置解读

以上配置指定了训练时需要的基本组件。例如：
- 训练轮数
- 采用什么损失函数去衡量性能
- 采用何种优化器去进行参数调优

---

## 五、其他配置

```yaml
val:
  val_freq: 1
  metric: psnr

checkpoint:
  save_freq: 10
  resume_from: null
  auto_resume: false
```

### 功能

上述简单的配置为训练引入了额外的功能：
- 训练时进行指定频次的验证，用来观察模型是否过拟合或者出现例如梯度爆炸等的意外情况
- 支持断点保存与重训

---

## 六、扩展你的模型

### 1.自定义 Dataset

1. 在 `uchiha/datasets/` 新增 `.py`
2. 定义自己的 Dataset 类，使用:

```python
from .builder import DATASET

@DATASET.register_module()
class MyDataset:
    ...
```

3. 在 `__init__.py` 中展示:

```python
from .my_dataset import MyDataset
```

### 2.自定义 Model

同理，在 `uchiha/models/` 中定义，使用:

```python
from .builder import MODEL

@MODEL.register_module()
class MyModel:
    ...
```

### 3.扩展其他组件

可扩展的组件包括：

* 优化器 (optimizer)
* 损失 (loss function)
* 学习率调度器 (scheduler)
* 表示功能单元 (metric module)

配合定制化配置文件，可以极大地模块化你的项目！

---

## 提示

* 更多 YAML 配置示例和经验请参考项目脚本

---

## 结论

通过配置文件加致构件实现，你可以无需频繁地编辑代码，快速进行调研和扩展。

支持模块化和维护性，为你的研究工作有效进行助力！
