import collections.abc
import random

import torch
from terminaltables import AsciiTable

from uchiha.utils import get_root_logger
from uchiha.utils.logger import print_log
from .metric import mean_absolute_error


def tensor2np(x):
    if isinstance(x, list):
        for idx, data in enumerate(x):
            x[idx] = tensor2np(data)
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return x


def regression_eval(preds, targets, elements, metric='MAE'):
    if isinstance(elements, collections.abc.Sequence):
        results = {element: 0 for element in elements}
    else:
        results = {elements: 0}

    preds = tensor2np(preds)
    targets = tensor2np(targets)
    for pred, target in zip(preds, targets):
        for j in range(len(elements)):
            metric_value = compute_metric(pred[:, j], target[:, j], metric)
            results[elements[j]] += metric_value

    for element in elements:
        results[element] /= len(preds)

    print_metrics(results)

    return results


def compute_metric(pred, target, metric='MAE'):
    if metric == 'MAE':
        return mean_absolute_error(pred, target)
    else:
        raise NotImplementedError(f'metric:{metric} is not supported yet')


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
