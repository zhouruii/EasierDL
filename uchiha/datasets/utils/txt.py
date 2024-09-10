import torch


def read_txt(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(torch.tensor([float(num) for num in line.strip().split(' ')]))

    return data
