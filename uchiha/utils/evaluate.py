import numpy as np
import torch
from terminaltables import AsciiTable

from uchiha.utils import get_root_logger
from uchiha.utils.logger import print_log
from .metric import mean_absolute_error, r2_score


def tensor2np(x):
    """ tensor --> ndarray

    Args:
        x (Tensor): Input tensor data

    Returns:
        ndarray: converted ndarray data
    """
    if isinstance(x, list):
        for idx, data in enumerate(x):
            x[idx] = tensor2np(data)
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return x


def regression_eval(preds, targets, elements, metric='MAE'):
    """ evaluation of regression tasks

    Args:
        preds (Tensor): Prediction of the model
        targets (Tensor): GT data
        elements (List[str]): Elements that need to be predicted
        metric (str): Evaluation metric. Default: MAE.

    Returns:
        dict: evaluation results for each element
    """

    results = {element: 0.0 for element in elements}


    preds = np.vstack(tensor2np(preds))
    targets = np.vstack(tensor2np(targets))
    metrics = compute_metric(preds, targets, metric)
    if len(results) == 1:
        results[elements[0]] = metrics
    else:
        for idx, element in enumerate(elements):
            results[element] = metrics[idx]

    print_metrics(results, metric)

    return results


def compute_metric(pred, target, metric='MAE'):
    """ given metric, calculation result

    Args:
        pred (ndarray): Prediction
        target (ndarray): GT
        metric (str): Evaluation metric. Default: MAE.

    Returns:
        float: Calculation results
    """
    if metric == 'MAE':
        return mean_absolute_error(pred, target)
    elif metric == 'R2':
        return 1 - (np.sum((target - pred) ** 2) / np.sum((target - np.mean(target)) ** 2))
        # return (np.sum((pred - np.mean(target)) ** 2) / np.sum((target - np.mean(target)) ** 2))
    else:
        raise NotImplementedError(f'metric:{metric} is not supported yet')


def print_metrics(result, metric):
    """ print metrics in the console

    Args:
        result (dict): Evaluation results for each element .
        metric (str): Evaluation metric.

    """
    header = ['Element', f'{metric}']
    table_data = [header]
    for key, value in result.items():
        row_data = [key, f'{value:.6f}']
        table_data.append(row_data)

    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    table.justify_columns[1] = 'center'

    logger = get_root_logger()
    print_log('\n' + table.table, logger=logger)


