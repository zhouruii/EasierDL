import numpy as np
import open3d as o3d
import plotly.graph_objects as go


def generate_3d_rain(height, width, depth, num_drops=10000, streak_length=20, wind_angle=135, wind_strength=10):
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
            xi = int(start_x + i * wind_dx * wind_strength / depth)  #
            yi = int(start_y + i * wind_dy * wind_strength / depth)  #
            zi = int(start_z - i)  # 雨条纹沿z轴向下移动

            if 0 <= xi < width and 0 <= yi < height and 0 <= zi < depth:
                x.append(xi)
                y.append(yi)
                z.append(zi)

    return x, y, z


def visualize_3d_rain_interactive(x, y, z):
    """
    使用Plotly动态可视化3D雨条纹模型。
    Args:
        x, y, z: 3D降雨的坐标点。
    """
    # 创建3D散点图
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=1,  # 点的大小
            color=z,  # 使用z轴值进行着色
            colorscale='Blues',  # 蓝色调的颜色映射
            opacity=0.8
        )
    )])

    # 设置图形布局
    fig.update_layout(
        title="Interactive 3D Rain Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        margin=dict(l=0, r=0, b=0, t=40)  # 减少边距以增加视图区域
    )

    # 显示图形
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
    height, width, depth = 512, 512, 100
    num_drops = 5000
    streak_length = 30
    wind_angle = 0
    wind_strength = 0

    x, y, z = generate_3d_rain(height, width, depth, num_drops, streak_length, wind_angle, wind_strength)

    # 使用Plotly进行交互式可视化
    visualize_3d_rain_interactive(x, y, z)
    # save_ply(x, y, z)
