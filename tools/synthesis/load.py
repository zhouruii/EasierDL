import rasterio
import spectral as sp


def read_data(path):
    with rasterio.open(path) as src:
        # 读取数据
        data = src.read()  # 数据形状为 (波段数, 行数, 列数)
        print("数据形状:", data.shape)

        # 读取元数据
        print("元数据:", src.meta)


def read_hdr(path):
    data = sp.open_image(path)

    bands_centers = data.bands.centers
    num_bands = data.nbands
    data_shape = data.shape
    gsd = data.metadata['map info'][5]
    print(num_bands, data_shape, gsd)

    return bands_centers, num_bands, data_shape, gsd


if __name__ == '__main__':
    # file_path = r"E:\datasets\ang20191021t151200_rfl_v2x1\ang20191021t151200_corr_v2x1_img"
    # read_data(file_path)
    file_path = r"E:\datasets\ang20191021t151200_rfl_v2x1\ang20191021t151200_corr_v2x1_img.hdr"
    read_hdr(file_path)
