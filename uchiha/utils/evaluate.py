import random

import torch
from terminaltables import AsciiTable

from uchiha.utils import get_root_logger
from uchiha.utils.logger import print_log
from .metric import mean_absolute_error


def tensor2np(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return x


def evaluate(preds, targets, elements):
    result = {element: 0 for element in elements}

    for i in range(len(preds)):
        pred = tensor2np(preds[i])
        target = tensor2np(targets[i])
        for j in range(len(elements)):
            metric = mean_absolute_error(pred[:, j], target[:, j])
            result[elements[j]] += metric

    for element in elements:
        result[element] /= len(preds)

    print_metrics(result)

    return result


def print_metrics(result):

    header = ['Element', 'MAE']
    table_data = [header]
    for key, value in result.items():
        row_data = [key, f'{value:.6f}']
        table_data.append(row_data)

    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    table.justify_columns[1] = 'center'

    logger = get_root_logger()
    print_log('\n' + table.table, logger=logger)


if __name__ == '__main__':
    txt = 'dataset/demo.txt'
    with open(txt, 'w', encoding='utf-8') as f:
        for _ in range(70):
            f.write(f'{random.uniform(0, 100):.3f} {random.uniform(0, 100):.3f}')
            f.write(f'\n')
