import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

from config import RAIN_STREAK
from tools.synthesis.rain_3d import generate_3d_rain


def project_orthogonal(points, axis='xy'):
    """正交投影到指定平面"""
    if axis == 'xy':
        return points[:, :2]  # XY平面（正视图）
    elif axis == 'xz':
        return points[:, [0, 2]]  # XZ平面（俯视图）
    elif axis == 'yz':
        return points[:, 1:]  # YZ平面（侧视图）
    else:
        raise ValueError("Invalid axis. Choose 'xy', 'xz', or 'yz'.")


if __name__ == '__main__':
    rain_3d = generate_3d_rain(height=512, width=512, depth=512, **RAIN_STREAK[1])
    points = np.vstack(rain_3d).T

    # # 设置画布
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #
    # # 正视图（XY平面）
    # proj_xy = project_orthogonal(points, axis='xy')
    # axes[0].scatter(proj_xy[:, 0], proj_xy[:, 1], s=1, alpha=0.5)
    # axes[0].set_title('Front View (XY Plane)')
    # axes[0].set_xlabel('X')
    # axes[0].set_ylabel('Y')
    # axes[0].grid(True)
    #
    # # 俯视图（XZ平面）
    # proj_xz = project_orthogonal(points, axis='xz')
    # axes[1].scatter(proj_xz[:, 0], proj_xz[:, 1], s=1, alpha=0.5)
    # axes[1].set_title('Top View (XZ Plane)')
    # axes[1].set_xlabel('X')
    # axes[1].set_ylabel('Z')
    # axes[1].grid(True)
    #
    # # 侧视图（YZ平面）
    # proj_yz = project_orthogonal(points, axis='yz')
    # axes[2].scatter(proj_yz[:, 0], proj_yz[:, 1], s=1, alpha=0.5)
    # axes[2].set_title('Side View (YZ Plane)')
    # axes[2].set_xlabel('Y')
    # axes[2].set_ylabel('Z')
    # axes[2].grid(True)
    #
    # plt.tight_layout()
    # plt.show()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 俯视图（沿 Y 轴方向投影）
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_view_control().set_front([0, 0, 1])  # 调整视角
    vis.run()
    vis.capture_screen_image("top_view.png")
    vis.destroy_window()
