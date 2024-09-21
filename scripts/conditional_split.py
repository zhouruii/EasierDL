import os
import shutil
from os.path import join


def split_data(data, label):
    to_split = []
    with open(label, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            content = line.strip()
            som = content
            if float(som) < 3:
                to_split.append(idx + 1)

    to_split = to_split[:16]
    dst_label = label.replace('train', 'val')
    with open(dst_label, 'w', encoding="utf-8") as f:
        for idx in to_split:
            src = join(data, f'{idx}.npy')
            dst = join(data.replace('train', 'val'), f'{idx}.npy')
            shutil.move(src, dst)

            f.write(f'{lines[idx - 1]}')


def save(data,dst):
    data = os.listdir(data)
    data = sorted(data,key=lambda x: int(x.split('.')[0]))
    with open(dst, 'w',encoding="utf-8") as f:
        for d in data:
            idx = int(d.split('.')[0])
            f.write(f'{idx} ')



if __name__ == '__main__':
    data = '/home/disk1/ZR/PythonProjects/uchiha/data/spectral_10/train/reflectivity'
    label = '/home/disk1/ZR/PythonProjects/uchiha/data/spectral_10/train/gt.txt'
    split_data(data, label)

    # data = '/home/disk1/ZR/PythonProjects/uchiha/data/spectral_10/val/reflectivity'
    # dst = '/home/disk1/ZR/PythonProjects/uchiha/data/split.txt'
    # save(data,dst)
