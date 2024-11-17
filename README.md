# 个人学习工具

## 数据集
数据集的准备与组织可参考[数据准备](docs/data.md)

## 使用

### 训练一个模型
```shell
python main.py --config ${config file}
```
示例:
```shell
python main.py --config configs/spectral/Zn/dwt_channel.yaml
```

### 测试一个模型
```shell
python test.py --config ${config file} --checkpoint ${checkpoint file}
```
示例:
```shell
python test.py --config configs/spectral/Zn/dwt_channel.yaml --checkpoint your checkpoint
```