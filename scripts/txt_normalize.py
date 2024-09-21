import numpy as np


def normalize(path, dst_path):
    data = []
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data.append(float(line.strip()))
    data = np.array(data)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    with open(dst_path, 'w', encoding="utf-8") as f:
        for d in data:
            f.write(f'{str(d)}\n')


def standardize(path, dst_path):
    data = []
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data.append(float(line.strip()))
    data = np.array(data)
    data = (data - np.mean(data)) / np.std(data)

    with open(dst_path, 'w', encoding="utf-8") as f:
        for d in data:
            f.write(f'{str(d)}\n')


if __name__ == '__main__':
    path = 'data/spectral/train/labelZn.txt'
    dst_path = 'data/spectral/train/labelZn_standardize.txt'
    standardize(path, dst_path)
