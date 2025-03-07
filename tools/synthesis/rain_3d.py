import numpy as np
import open3d as o3d
import plotly.graph_objects as go


def generate_3d_rain(height, width, depth, num_drops=10000, streak_length=20, wind_angle=135, wind_strength=0.5,
                     streak_width=0.5):
    """
    生成3维降雨模型，包括雨条纹的方向和连续性。
    Args:
        height: 3D雨场的Y方向大小。
        width: 3D雨场的X方向大小。
        depth: 3D雨场的Z方向大小。
        num_drops: 雨条纹的数量。
        wind_angle: 风的方向，单位为度。
        wind_strength: 风的强度（影响雨条纹的倾斜程度）。
    Returns:
        x, y, z: 雨条纹的3D坐标。
    """
    x, y, z = [], [], []

    # 风向的单位向量
    wind_dx = np.sin(np.deg2rad(wind_angle))
    wind_dy = np.cos(np.deg2rad(wind_angle))

    for _ in range(num_drops):
        # 雨条纹的起始位置
        start_x = np.random.randint(0, width)
        start_y = np.random.randint(0, height)
        start_z = np.random.randint(0, depth)

        for i in range(streak_length):
            xi = int(start_x + i * wind_dx * wind_strength)  #
            yi = int(start_y + i * wind_dy * wind_strength)  #
            zi = int(start_z - i)  # 雨条纹沿z轴向下移动

            # x, y 加入随机噪声模拟雨滴宽度（横向散布）
            xi = xi + np.random.normal(0, streak_width)
            yi = yi + np.random.normal(0, streak_width)

            if 0 <= xi < width and 0 <= yi < height and 0 <= zi < depth:
                x.append(xi)
                y.append(yi)
                z.append(zi)

    return x, y, z


def visualize_3d_rain(x, y, z, intensity=None):
    """
    使用Plotly可视化3D降雨点云，强度值映射到颜色和点大小。
    Args:
        x, y, z: 雨条纹的3D坐标。
        intensity: 雨条纹的强度值。
    """
    # 创建3D散点图
    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=1,  # 基础点大小
            color=intensity if intensity else z,  # 强度值映射到颜色
            colorscale='Blues',  # 颜色映射方案
            opacity=0.8,  # 整体透明度
            colorbar=dict(title='Intensity')  # 颜色条
        )
    )

    # 创建布局
    layout = go.Layout(
        title='3D Rain Streaks Visualization',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # 创建图形并显示
    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()


def save_ply(x, y, z, filename='output.ply'):
    # 将x, y, z列表转换为一个N×3的NumPy数组
    points = np.vstack((x, y, z)).T  # 形成一个形状为(N, 3)的数组

    # 创建一个Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()

    # 将点云的坐标赋值给点云对象
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 将点云保存为ply文件
    o3d.io.write_point_cloud(filename, point_cloud)
    print(f"Point cloud saved to {filename}")


if __name__ == '__main__':
    # 生成3D雨模型
    height, width, depth = 512, 512, 512
    num_drops = 3000
    streak_length = 40
    wind_angle = 60
    wind_strength = 0.2
    streak_width = 1.5

    x, y, z = generate_3d_rain(height, width, depth, num_drops, streak_length, wind_angle, wind_strength, streak_width)

    visualize_3d_rain(x, y, z)
