import os

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def kde(data):
    plt.figure()  # 创建一个新图
    plt.title(f'Zn_KDE')

    for i, row in enumerate(data):
        sns.kdeplot(row, shade=True)
        plt.xlabel('Value')
        plt.ylabel('Density')

    plt.savefig('data/kde_Zn.png', dpi=300)


if __name__ == '__main__':
    # data_path = 'data/spectral/val/reflectivity'
    # data = []
    # for name in os.listdir(data_path):
    #     path = os.path.join(data_path, name)
    #     tmp = np.load(path)[1, 1, :]
    #     data.append(tmp)

    gt_path = 'data/Zn.txt'
    data = []
    with open(gt_path, 'r',encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data.append(float(line.strip()))

    data = [data]

    kde(data)
