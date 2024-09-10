import torch

from uchiha.utils import evaluate


def validate(epoch, dataloader, model, writer):
    dataset = dataloader.dataset
    elements = dataset.ELEMENTS
    targets = []
    preds = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            # data
            spectral_data = data['spectral_data'].cuda()
            targets.append(data['target'])

            # forward
            with torch.no_grad():
                pred = model(spectral_data).squeeze()
                preds.append(pred)

    # evaluate
    result = evaluate(preds, targets, elements)

    # log
    for i in range(len(elements)):
        writer.add_scalar(f'{elements[i]}_metric', result[elements[i]], epoch)

    return result
