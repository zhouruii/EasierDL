import torch


def validate(epoch, dataloader, model, writer, metric):
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
