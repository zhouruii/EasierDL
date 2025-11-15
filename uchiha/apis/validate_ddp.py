import torch
import torch.distributed as dist


def validate_ddp(epoch, dataloader, model, writer, metric, device, rank, world_size):
    """ validation during training

    Args:
        epoch (int): the training epoch when the validation frequency is reached
        dataloader (torch.utils.data.Dataloader): validation set's dataloader
        model (torch.nn.Module): model built from configuration file
        writer (SummaryWriter): tensorboard-based loggers
        metric (str): model's performance metrics for specific tasks
        device (torch.device): device to run the model

    Returns:
        results (dict): Dictionary, the key is the name corresponding to cls or ele,
        and the value is the result of the model output.

    """
    dataset = dataloader.dataset
    model.eval()

    # local storage per rank
    local_preds = []
    local_targets = []
    local_indexes = []

    with torch.no_grad():
        for data in dataloader:
            sample = data['sample'].to(device, non_blocking=True)
            pred = model(sample)

            local_preds.append(pred.cpu().numpy())
            local_targets.append(data['target'].numpy())
            local_indexes.append(data['index'].numpy())

    # gather lists from all ranks
    all_preds = [None for _ in range(world_size)]
    all_targets = [None for _ in range(world_size)]
    all_indexes = [None for _ in range(world_size)]

    dist.all_gather_object(all_preds, local_preds)
    dist.all_gather_object(all_targets, local_targets)
    dist.all_gather_object(all_indexes, local_indexes)

    # only rank 0 runs evaluation
    if rank == 0:
        # flatten list-of-list
        preds_flat = [x for plist in all_preds for x in plist]
        targets_flat = [x for plist in all_targets for x in plist]
        indexes_flat = [x for plist in all_indexes for x in plist]

        results = dataset.evaluate(preds_flat, targets_flat, metric, indexes_flat)

        # log results
        if isinstance(results, (int, float)):
            writer.add_scalar(metric, results, epoch)
        elif isinstance(results, dict):
            for k, v in results.items():
                writer.add_scalar(k, v, epoch)
        return results

    else:
        # non-zero ranks return nothing
        return None
