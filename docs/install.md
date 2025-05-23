### 1.克隆本仓库
```shell
git clone https://github.com/zhouruii/uchiha.git
cd uchiha-main
```

### 2.创建虚拟环境
`conda create -n HDRformer python=3.9`

### 3.安装torch
注意pytorch与CUDA版本的选择,参考[pytorch官网](https://pytorch.org/)

`pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111`

### 4.安装依赖
`pip install -r requirements.txt`