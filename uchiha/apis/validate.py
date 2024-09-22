import torch


def validate(epoch, dataloader, model, writer, metric):
    """ validation during training

    Args:
        epoch (int): the training epoch when the validation frequency is reached
        dataloader (torch.utils.data.Dataloader): validation set's dataloader
        model (torch.nn.Module): model built from configuration file
        writer (SummaryWriter): tensorboard-based loggers
        metric (str): model's performance metrics for specific tasks

    Returns:
        results (dict): Dictionary, the key is the name corresponding to cls or ele,
        and the value is the result of the model output.

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
