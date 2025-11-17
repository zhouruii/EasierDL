from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from uchiha.utils import get_root_logger


def simple_test(dataloader, model, metric):
    """ simple test

    Args:
        dataloader (torch.utils.data.Dataloader): validation set's dataloader
        model (torch.nn.Module): model built from configuration file
        metric (str): model's performance metrics for specific tasks

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


def calculate_psnr(labels, outputs):
    assert labels.shape == outputs.shape, "Shapes of labels and outputs must be the same"

    # 转换数据类型为 float
    labels = labels.float()
    outputs = outputs.float()
    # 初始化 SSIM
    psnr_total = 0
    max_val = 1.0  # 假设像素值的范围是 [0, 1]

    # 遍历每个通道
    for i in range(labels.shape[1]):
        mse = torch.mean((labels[:, i, :, :] - outputs[:, i, :, :]) ** 2)
        psnr_channel = 20 * torch.log10(max_val / torch.sqrt(mse))
        psnr_channel = min(psnr_channel, 100)
        psnr_total += psnr_channel

    psnr_avg = psnr_total / labels.shape[1]

    return psnr_avg


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel).cpu().numpy()
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel).cpu().numpy()
    # print(type(mu2))
    mu1_sq = np.power(mu1, 2)  # mul的2次方
    mu2_sq = np.power(mu2, 2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel).cpu().numpy() - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel).cpu().numpy() - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel).cpu().numpy() - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return np.mean(ssim_map)
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)  # 将输入input张量每个元素的夹紧到区间
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def calculate_uqi(labels, outputs):
    assert labels.shape == outputs.shape, "Shapes of labels and outputs must be the same"

    # 转换数据类型为 float
    labels = labels.float()
    outputs = outputs.float()
    labels = labels.cpu().numpy()
    outputs = outputs.cpu().numpy()
    # labels = labels.numpy()
    # outputs = outputs.numpy()
    s, c, h, w = labels.shape

    uqi_total = 0
    for i in range(labels.shape[1]):
        output = outputs[:, i, :, :]
        output = np.reshape(output, (1, h * w))
        output = output.squeeze()
        label = labels[:, i, :, :]
        label = np.reshape(label, (1, h * w))
        label = label.squeeze()
        mean1 = np.mean(output)
        mean2 = np.mean(label)
        var1 = np.var(output)
        var2 = np.var(label)
        # cov = np.cov(output, label, mean1, mean2)
        cov = np.sum((output - mean1) * (label - mean2)) / (h * w - 1)

        uqi_channel = 4 * mean1 * mean2 * cov / (
            ((mean1 ** 2 + mean2 ** 2) * (var1 + var2)).clip(0.0000000001, 1000000))
        # uqi_channel = ssim(outputs[:, i, :, :], labels[:, i, :, :])
        # 将每个通道的 SSIM 累加
        uqi_total += uqi_channel

    # 对所有通道的 SSIM 求平均
    uqi_avg = uqi_total / labels.shape[1]

    return uqi_avg


def calculate_sam(labels, outputs):
    assert labels.shape == outputs.shape, "Shapes of labels and outputs must be the same"

    # 转换数据类型为 float
    labels = labels.float()
    outputs = outputs.float()
    labels = labels.cpu().numpy()
    outputs = outputs.cpu().numpy()

    s, c, h, w = labels.shape
    labels = np.reshape(labels, (c, h * w))
    labels = labels.transpose(1, 0)
    outputs = np.reshape(outputs, (c, h * w))
    outputs = outputs.transpose(1, 0)
    core = np.multiply(labels, outputs)
    mole = np.sum(core, axis=1)
    ln = np.sqrt(np.sum(np.square(labels), axis=1))
    on = np.sqrt(np.sum(np.square(outputs), axis=1))
    deno = np.multiply(ln, on)
    deno = np.clip(deno, 0.000000001, np.max(deno))
    # print(mole,"123",deno,"123",core,"123",ln,"123",on)
    sam = np.rad2deg(np.arccos((mole / deno).clip(-1, 1)))
    # print(np.mean(mole/deno))
    return np.mean(sam)


def hsi_test(dataloader, model, device, no_reference=False):
    """ test for HSI

    Args:
        dataloader (torch.utils.data.Dataloader): validation set's dataloader
        model (torch.nn.Module): model built from configuration file
        device (torch.device): device to run the model
        no_reference(bool): no reference metric: NIQE AG

    """
    logger = get_root_logger()
    logger.info('start evaluating...')

    psnrs = []
    ssims = []
    uqis = []
    sams = []
    niqes = []
    ags = []
    model.eval()

    pbar = tqdm(dataloader, desc="testing", total=len(dataloader))

    with torch.no_grad():
        for idx, data in enumerate(pbar):
            # data
            sample = data['sample'].to(device, non_blocking=True)
            target = data['target'].to(device, non_blocking=True)
            # forward
            pred = model(sample)
            # test
            psnrs.append(calculate_psnr(target, pred).item())
            ssims.append(calculate_ssim(target, pred))
            uqis.append(calculate_uqi(target, pred))
            sams.append(calculate_sam(target, pred))
            # if no_reference:
            #     niqes.append(calculate_niqe(pred))
            #     ags.append(calculate_ag(pred))

            pbar.set_postfix(iter=f"{idx + 1}/{len(dataloader)}")

    pbar.close()

    logger.info(f'Mean PSNR: {np.mean(psnrs):.4f} dB')
    logger.info(f'Mean SSIM: {np.mean(ssims):.4f}')
    logger.info(f'Mean UQI: {np.mean(uqis):.4f}')
    logger.info(f'Mean SAM: {np.mean(sams):.4f}')
    if no_reference:
        logger.info(f'Mean NIQE: {np.mean(niqes):.4f}')
        logger.info(f'Mean AG: {np.mean(ags):.4f}')
