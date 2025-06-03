import torch
from tqdm import tqdm


def simple_test(dataloader, model, metric):
    """ validation during training

    Args:
        dataloader (torch.utils.data.Dataloader): validation set's dataloader
        model (torch.nn.Module): model built from configuration file
        metric (str): model's performance metrics for specific tasks

    Returns:
        results (dict): Dictionary, the key is the name corresponding to cls or ele,
        and the value is the result of the model output.

    """
    dataset = dataloader.dataset
    targets = []
    preds = []
    indexes = []
    model.eval()

    pbar = tqdm(dataloader, desc="testing", total=len(dataloader))

    with torch.no_grad():
        for idx, data in enumerate(pbar):
            # data
            sample = data['sample'].cuda()
            # forward
            pred = model(sample)

            targets.append(data['target'].numpy())
            indexes.append(data['index'].numpy())
            preds.append(pred.cpu().numpy())

            pbar.set_postfix(iter=f"{idx + 1}/{len(dataloader)}")

    # evaluate
    results = dataset.evaluate(preds, targets, metric, indexes)

    pbar.close()

    return results
