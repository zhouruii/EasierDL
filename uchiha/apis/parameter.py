import numpy as np
from tabulate import tabulate


def count_parameters(module):
    """ Calculate the parameters of the model.

    Args:
        module (nn.Module): The model that require counting parameter quantities

    Returns:
        int: Parameter quantities of the model
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def analyze_model(model, depth=0, max_depth=3, parent_name="", visited_params=None):
    results = []
    if depth > max_depth:
        return results

    # 初始化已统计参数集合
    if visited_params is None:
        visited_params = set()

    for name, child in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        # 统计当前模块的独立参数（排除已统计过的参数）
        child_params = set(id(p) for p in child.parameters())
        new_params = child_params - visited_params
        param_count = sum(p.numel() for p in child.parameters()
                          if p.requires_grad and id(p) in new_params)

        # 更新已统计参数集合
        visited_params.update(new_params)

        module_info = {
            "name": full_name,
            "type": child.__class__.__name__,
            "params": param_count,
            "depth": depth,
            "is_leaf": len(list(child.children())) == 0
        }
        results.append(module_info)

        # 递归分析子模块（传递已统计参数集合）
        if not module_info["is_leaf"]:
            results.extend(analyze_model(
                child, depth + 1, max_depth, full_name, visited_params
            ))

    return results


def format_size(num_bytes):
    """格式化参数量显示"""
    if num_bytes == 0:
        return "0"
    units = ['', 'K', 'M', 'B', 'T']
    k = 1000.0
    magnitude = int(np.floor(np.log(num_bytes) / np.log(k)))
    return f"{num_bytes / k ** magnitude:.2f}{units[magnitude]}"


def visualize_analysis(results, logger):
    """可视化展示分析结果"""
    table_data = []
    total_params = 0

    for item in results:
        indent = "    " * item["depth"]
        display_name = f"{indent}└─ {item['name']}" if item["depth"] > 0 else item["name"]

        # 计算百分比
        param_count = item["params"]
        total_params += param_count

        table_data.append([
            display_name,
            item["type"],
            format_size(param_count),
            f"{param_count:,}",
            "✓" if item["is_leaf"] else ""
        ])

    # 按参数量降序排序
    table_data.sort(key=lambda x: -int(x[3].replace(",", "")))

    # 添加总计行
    table_data.append([
        "TOTAL",
        "",
        format_size(total_params),
        f"{total_params:,}",
        ""
    ])

    logger.info(tabulate(
        table_data,
        headers=["Module", "Type", "Params (Formatted)", "Params (Exact)", "Leaf"],
        tablefmt="grid",
        stralign="right"
    ))

    return total_params


def analyze_model_structure(model, logger, max_depth=3):
    if max_depth == 0:
        return
    """完整分析流程"""
    logger.info(f"\n{' Model Analysis ':=^80}")
    logger.info(f"Model Class: {model.__class__.__name__}")
    results = analyze_model(model, max_depth=max_depth)
    total = visualize_analysis(results, logger)

    # 打印内存占用估算
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    logger.info(f"\nEstimated Memory Usage: {format_size(param_size + buffer_size)} bytes")
    logger.info(f"Trainable Parameters: {format_size(total)}")
    logger.info("=" * 80)

    return total


if __name__ == '__main__':
    print(format_size(1401924))
