import numpy as np
import spectral
import plotly.graph_objects as go


def visualize_3d_cube(data, rgb_indices):
    x, y, z = np.indices(data.shape[:3])
    values = data[:, :, rgb_indices].reshape(-1, 3)

    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=np.linalg.norm(values, axis=1),
        colorscale='Viridis',
        opacity=0.2,
        surface_count=50
    ))
    fig.update_layout(scene=dict(xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False))

    # 保存为 HTML（交互式）
    fig.write_html("3d_visual.html")

    # 保存为 PNG（静态图像）
    fig.write_image("3d_plot.png", width=800, height=600)


if __name__ == "__main__":
    lq_path = '/home/disk2/ZR/datasets/AVIRIS/512/rain/storm/f130804t01p00r04rdn_e_23.npy'
    gt_path = '/home/disk2/ZR/datasets/AVIRIS/512/gt/f130804t01p00r04rdn_e_23.npy'
    clean = np.load(gt_path)
    noisy = np.load(lq_path)

    _bands = [36, 19, 8]
    # bands = (136, 67, 18)

    spectral.view_cube(noisy, bands=_bands)
    # visualize_3d_cube(noisy, rgb_indices=_bands)
