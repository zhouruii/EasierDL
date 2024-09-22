import torch


def validate(epoch, dataloader, model, writer, metric):
    """ 训练时的验证

    Args:
        epoch (int): 到达验证频次时的训练轮数
        dataloader (torch.utils.data.Dataloader): 构建好的验证数据加载器
        model (torch.nn.Module): 构建好的模型
        writer (SummaryWriter): 基于tensorboard的记录器
        metric (str): 检测模型对特定任务的性能指标

    Returns:
        results (dict): 字典，键为cls或ele对应的名称，值为模型输出的结果

    """
    dataset = dataloader.dataset
    targets = []
    preds = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            # data
            sample = data['sample'].cuda()
            targets.append(data['target'])

            # forward
            with torch.no_grad():
                pred = model(sample)
                preds.append(pred)

    # evaluate
    results = dataset.evaluate(preds, targets, metric)

    # log
    for ele in results:
        writer.add_scalar(f'{ele}_{metric}', results[ele], epoch)

    return results
