from ..utils import print_log, get_root_logger


def train_by_epoch(epoch, dataloader, model, optimizer, scheduler, criterion, writer):
    model.train()
    for idx, data in enumerate(dataloader):
        # data
        spectral_data = data['spectral_data'].cuda()
        target = data['target'].cuda().float()

        # forward & loss
        pred = model(spectral_data)
        loss = criterion(pred, target)

        # backward & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        if (idx + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print_log(f'epoch:[{epoch + 1}], iter:[{idx + 1}/{len(dataloader)}], loss: {loss}, lr:{current_lr}',
                      get_root_logger())

        writer.add_scalar('loss', loss.item(), epoch * len(dataloader) + idx)

    scheduler.step()

    return writer, model, optimizer, scheduler
