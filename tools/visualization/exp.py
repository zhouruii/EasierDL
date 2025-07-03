import numpy as np
import matplotlib.pyplot as plt
import torch


def process_image_np(image: np.ndarray, split_points: list) -> np.ndarray:
    """
    输入: HWC ndarray
    功能: 排序 + 切分点差值
    输出: HWN ndarray
    """
    h, w, c = image.shape
    flattened = image.reshape(-1, c)
    sorted_flattened = np.sort(flattened, axis=1)

    results = []
    for split_idx in split_points:
        if not (0 < split_idx < c):
            raise ValueError(f"切分点 {split_idx} 超出通道范围 (1~{c-1})")
        front = sorted_flattened[:, :split_idx]
        back = sorted_flattened[:, split_idx:]
        diff = np.mean(front, axis=1) - np.mean(back, axis=1)
        results.append(diff)

    output = np.stack(results, axis=1)
    return output.reshape(h, w, -1)


def save_channels_np(image: np.ndarray, prefix: str = "output_channel"):
    """
    对 HWN ndarray 保存每个通道图像
    """
    h, w, n = image.shape
    for i in range(n):
        plt.imshow(image[:, :, i], cmap='gray')
        plt.title(f"Channel {i}")
        plt.colorbar()
        filename = f"{prefix}_{i}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")


def process_image_torch(image: torch.Tensor, split_points: list) -> torch.Tensor:
    """
    输入: CHW torch.Tensor
    功能: 排序 + 切分点差值
    输出: HWN torch.Tensor
    """
    c, h, w = image.shape
    flattened = image.permute(1, 2, 0).reshape(-1, c)  # L x C
    sorted_flattened, _ = torch.sort(flattened, dim=1)

    results = []
    for split_idx in split_points:
        if not (0 < split_idx < c):
            raise ValueError(f"切分点 {split_idx} 超出通道范围 (1~{c-1})")
        front = sorted_flattened[:, :split_idx]
        back = sorted_flattened[:, split_idx:]
        diff = front.mean(dim=1) - back.mean(dim=1)
        results.append(diff)

    output = torch.stack(results, dim=1)  # L x N
    return output.reshape(h, w, -1)  # HWN


if __name__ == "__main__":
    # NumPy 测试
    H, W, C = 100, 100, 10
    dummy_image = np.random.rand(H, W, C)
    splits = [C // 2, C // 3]
    out_np = process_image_np(dummy_image, splits)
    print(f"NumPy 输出形状: {out_np.shape}")
    save_channels_np(out_np, prefix="np_channel")

    # PyTorch 测试
    dummy_tensor = torch.rand(C, H, W)
    out_torch = process_image_torch(dummy_tensor, splits)
    print(f"PyTorch 输出形状: {out_torch.shape}")
