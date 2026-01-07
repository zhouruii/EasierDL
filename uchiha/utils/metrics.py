import cv2
import numpy as np
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
import scipy
from PIL import Image
import scipy.ndimage
import scipy.special
import math
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

import skimage.measure

from uchiha.utils.data import normalize


def calculate_psnr_ssim(target, pred):
    """
    计算 PSNR 和 SSIM
    Args:
        target (ndarray): HWC 形状, 真实值
        pred (ndarray): HWC 形状, 预测值
    Returns:
        psnr (float)
        ssim (float)
    """

    # 1. 计算 PSNR
    data_range = target.max() - target.min()
    psnr_val = psnr_func(target, pred, data_range=data_range)

    # 2. 计算 SSIM
    # channel_axis=2 表示通道在最后一个维度 (H, W, C)
    ssim_val = ssim_func(target, pred, data_range=data_range, channel_axis=2)

    return psnr_val, ssim_val


def calculate_uqi(target, pred):
    """
    计算 UQI (Universal Quality Image Index)
    UQI 是 SSIM 的一种特殊情况（不含亮度/对比度常数），
    通常对每个波段计算后取平均。
    """

    def _uqi_single_channel(t, p):
        # 展平以便计算统计量
        t = t.flatten()
        p = p.flatten()

        mx = np.mean(t)
        my = np.mean(p)

        # 样本协方差与方差 (使用 N-1 或 N 均可，保持一致即可，这里用 numpy 默认)
        cov_xy = np.cov(t, p)[0][1]
        var_x = np.var(t)
        var_y = np.var(p)

        # 避免分母为 0
        eps = 1e-8

        # UQI 公式
        numerator = 4 * cov_xy * mx * my
        denominator = (var_x + var_y) * (mx ** 2 + my ** 2) + eps

        return numerator / denominator

    # 逐通道计算
    channels = target.shape[2]
    uqis = []
    for i in range(channels):
        uqis.append(_uqi_single_channel(target[:, :, i], pred[:, :, i]))

    return np.mean(uqis)


def calculate_sam(target, pred):
    """
    计算 SAM (Spectral Angle Mapper)
    Args:
        target: (H, W, C)
        pred: (H, W, C)
    Returns:
        mean_sam (float): 平均光谱角（单位：弧度 rad）
    """
    # 确保没有 0 向量，避免除以 0
    eps = 1e-8

    # 在通道维度 (axis=2) 上计算点积
    # dot_product shape: (H, W)
    dot_product = np.sum(target * pred, axis=2)

    # 计算范数 (L2 norm)
    # norm shape: (H, W)
    norm_target = np.linalg.norm(target, axis=2)
    norm_pred = np.linalg.norm(pred, axis=2)

    # 计算余弦值
    denominator = norm_target * norm_pred + eps
    cos_theta = dot_product / denominator

    # 截断数值以防数值误差导致 arccos 越界
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 计算角度 (arccos 得到的是弧度)
    sam_map = np.arccos(cos_theta)

    # 如果有的像素全是0，sam_map 可能是 nan，将其置为 0
    sam_map = np.nan_to_num(sam_map)

    return np.mean(sam_map)


def calculate_ag(img):
    """
    计算 AG (Average Gradient) - 无参考指标
    衡量图像的清晰度/纹理丰富程度
    """
    # img shape: (H, W, C)
    # 计算 x 和 y 方向的梯度
    # 使用简单的差分或者是 Sobel 算子。AG 标准定义常用简单的差分。

    # 逐通道计算，最后取平均
    img = normalize(img)
    channels = img.shape[2]
    ag_vals = []

    for c in range(channels):
        band = img[:, :, c]
        sobelx = cv2.Sobel(band, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(band, cv2.CV_64F, 0, 1, ksize=3)
        ag_val = np.mean(np.sqrt(sobelx ** 2 + sobely ** 2))
        ag_vals.append(ag_val)

    return np.mean(ag_vals)


gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0 / gamma_range)
a *= a
b = scipy.special.gamma(1.0 / gamma_range)
c = scipy.special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)


def aggd_features(imdata):
    # flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
        gamma_hat = left_mean_sqrt / right_mean_sqrt
    else:
        gamma_hat = np.inf
    # solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata)) ** 2) / (np.average(imdata2))
    else:
        r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    # solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm) ** 2);
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    # mean parameter
    N = (br - bl) * (gam2 / gam1)  # *aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)


def ggd_features(imdata):
    nr_gam = 1 / prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq / E ** 2
    pos = np.argmin(np.abs(nr_gam - rho));
    return gamma_range[pos], sigma_sq


def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights


def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image ** 2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image ** 2))
    return (image - mu_image) / (var_image + C), var_image, mu_image


def _niqe_extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl + br) / 2.0,
                     alpha1, N1, bl1, br1,  # (V)
                     alpha2, N2, bl2, br2,  # (H)
                     alpha3, N3, bl3, bl3,  # (D1)
                     alpha4, N4, bl4, bl4,  # (D2)
                     ])


def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)


def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)


def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = np.int(patch_size)
    patches = []
    for j in range(0, h - patch_size + 1, patch_size):
        for i in range(0, w - patch_size + 1, patch_size):
            patch = img[j:j + patch_size, i:i + patch_size]
            patches.append(patch)

    patches = np.array(patches)

    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features


def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)

    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)
    img2 = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)

    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size / 2)

    feats = np.hstack((feats_lvl1, feats_lvl2))  # feats_lvl3))

    return feats


def calculate_niqe(inputImgData):
    inputImgData = normalize(inputImgData)
    patch_size = 96
    module_path = dirname(__file__)

    # TODO: memoize
    params = scipy.io.loadmat(join(module_path, 'niqe_image_params.mat'))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]

    M, N, C = inputImgData.shape

    # assert C == 1, "niqe called with videos containing %d channels. Please supply only the luminance channel" % (C,)
    assert M > (
            patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert N > (
            patch_size * 2 + 1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"

    if inputImgData.max() <= 1.1:
        inputImgData = inputImgData * 255.0
    niqe_list = []

    for c in range(C):
        channel_data = inputImgData[:,:,c]
        feats = get_patches_test_features(channel_data, patch_size)
        sample_mu = np.mean(feats, axis=0)
        sample_cov = np.cov(feats.T)

        X = sample_mu - pop_mu
        covmat = ((pop_cov + sample_cov) / 2.0)
        pinvmat = scipy.linalg.pinv(covmat)
        niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))
        niqe_list.append(niqe_score)

    return np.mean(niqe_list)


