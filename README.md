# 🌟 深度学习简易工具箱 | Uchiha

> **DIY your model!**  
> 快速搭建、自由配置多种经典与创新的深度学习模型结构。

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8.2-orange)](https://pytorch.org/)

---

## 📚 目录
- [简介](#简介)
- [Demo](#demo)
- [安装](#安装)
- [使用](#使用)
- [数据集](#数据集)
- [更多文档](#更多文档)

---

## 📖 简介

通过**配置文件**，来自定义你的模型！  
支持多种基础模型结构：
- 串行结构（如经典的深度 CNN）
- 并行结构
- 知名的 U-Net 结构 等等

<div align="center">
  <img src="docs/assets/jpg/stack.jpg" alt="串行结构" width="400"/>
</div>
<div align="center">

  <img src="docs/assets/jpg/parallel.jpg" alt="并行结构" width="400"/>
</div>

---

## 🚀 Demo

```shell
python main.py --config ${config file}
```

👉 关于详细的配置说明以及各类模块参数，可以参见[项目文档](https://zhouruii.github.io/uchiha/)

---

## ⚙️ 安装

### 1️⃣ 克隆仓库

```shell
git clone https://github.com/zhouruii/uchiha.git
cd uchiha-main
```

### 2️⃣ 创建虚拟环境

```shell
conda create -n your_env_name python=3.9
conda activate your_env_name
```

### 3️⃣ 安装 PyTorch

⚠️ 请根据你的 CUDA 版本选择合适的 PyTorch，参考 [PyTorch 官网](https://pytorch.org/)

```shell
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
```

### 4️⃣ 安装依赖

```shell
pip install -r requirements.txt
```

---

## 🏃‍♂️ 使用

### 常用参数

| 参数 | 描述 |
| ------ | ------ |
| `--seed` | 随机数种子 |
| `--config` | 训练程序的配置文件（核心） |
| `--gpu_ids` | 显卡 ID，支持多卡 |
| `--analyze_params` | 参数分析深度（0=总参数量） |

📄 更多详情：[配置说明](docs/config.md)

---

### 🚦 训练示例

```shell
python main.py --config ${config file}
```

多卡训练 + 参数递归分析：

```shell
python main.py --analyze_params 3 --gpu_ids 0 1 2 3 --config configs/baseline/Restormer.yaml
```

---

### 🔍 测试示例

```shell
python test.py --config ${config file} --checkpoint ${checkpoint file}
```

示例：

```shell
python test.py --config configs/baseline/Restormer.yaml --checkpoint your_checkpoint
```

---

## 📂 数据集

数据集准备与组织请参考 [数据准备](docs/data.md)

---

## 🔗 更多文档

- [配置文件说明](docs/config.md)
- [数据集准备](docs/data.md)

---

**Have fun & happy training! 🚀**
