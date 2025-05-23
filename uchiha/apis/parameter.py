from tabulate import tabulate
from torch import nn


def count_parameters(module):
    """ Calculate the parameters of the model.

    Args:
        module (nn.Module): The model that require counting parameter quantities

    Returns:
        int: Parameter quantities of the model
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def unwrap_model(model):
    """解包DDP/DataParallel包裹的模型"""
    if hasattr(model, 'module'):  # 处理DataParallel/DDP包裹
        return model.module
    return model


def format_param_count(n):
    """格式化参数量为 K/M 单位。"""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K"
    else:
        return str(n)


def log_model_parameters(model: nn.Module, logger, max_depth: int = 1):
    """
    记录模型的参数结构，以表格形式展示参数总数和比例，并按参数量排序。

    Args:
        model (nn.Module): 模型对象。
        logger: 日志记录器。
        max_depth (int): 展示的最大层级。
    """

    def collect_named_param_info(_model, _max_depth):
        _rows = []
        total_params = count_parameters(_model)
        stack = [("", _model, 0)]  # (路径, 模块, 当前深度)

        while stack:
            prefix, module, depth = stack.pop(0)
            if depth > _max_depth:
                continue

            name = prefix if prefix != "" else _model.__class__.__name__
            n_params = count_parameters(module)
            percent = f"{(n_params / total_params) * 100:.2f}%" if total_params > 0 else "0.00%"

            _rows.append({
                "depth": depth,
                "name": name,
                "type": module.__class__.__name__,
                "params": n_params,
                "percent": percent
            })

            if depth < _max_depth:
                for child_name, child in module.named_children():
                    full_name = f"{prefix}.{child_name}" if prefix else child_name
                    stack.append((full_name, child, depth + 1))

        return _rows, total_params

    logger.info("== Start analyzing model parameters ==")
    rows, total = collect_named_param_info(model, max_depth)

    # ✅ 按参数量降序排序
    rows.sort(key=lambda x: x["params"], reverse=True)

    # ✅ 转为 tabulate 可读格式
    table_data = [
        [r["depth"], r["name"], r["type"], format_param_count(r["params"]), r["percent"]]
        for r in rows
    ]

    table = tabulate(
        table_data,
        headers=["Depth", "Module Name", "Type", "Params", "Percent"],
        tablefmt="fancy_grid"
    )

    logger.info("\n" + table)
    logger.info(f"== Total parameters: {format_param_count(total)} ==")
