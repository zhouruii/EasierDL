# 数据集准备
在项目根目录下新建data目录来存放数据集

## 高光谱图像
data目录下的数据组织形式为：
```none
Project Root
├── uchiha
├── scripts
├── configs
├── data
│   ├── spectral
│   │   ├── train
│   │   │   │── reflectivity
│   │   │   │── GT.txt
│   │   ├── val
│   │   │   │── reflectivity
│   │   │   │── GT.txt
│   │   ├── trainval(optional)
│   │   │   │── reflectivity
│   │   │   │── GT.txt
│   │   ├── test
│   │   │   │── reflectivity
│   │   │   │── GT.txt
```