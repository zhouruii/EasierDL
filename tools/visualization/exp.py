import matplotlib.pyplot as plt

# 示例数据
hidden_channels = [32, 48, 64]  # MLP 的隐藏层通道数
psnr_data_1 = [74.4, 79.2, 84.7]        # 第一组 PSNR 数据
psnr_data_2 = [41.1, 42.2, 47.9]        # 第二组 PSNR 数据

# 创建折线图
plt.figure(figsize=(8, 5))

# 绘制第一组数据
plt.plot(hidden_channels, psnr_data_1, marker='o', linestyle='-', color='blue', label='Greyc')
# 在每个点上添加文本
for x, y in zip(hidden_channels, psnr_data_1):
    plt.text(x, y, f'{y}', fontsize=9, ha='right')

# 绘制第二组数据
plt.plot(hidden_channels, psnr_data_2, marker='s', linestyle='-', color='orange', label='8iVFBv2')
# 在每个点上添加文本
for x, y in zip(hidden_channels, psnr_data_2):
    plt.text(x, y, f'{y}', fontsize=9, ha='right')

# 设置坐标轴标签和标题
plt.xlabel('hidden layer in MLP')
plt.ylabel('PSNR')
plt.title('')

# 自定义图例
plt.legend(loc='upper left')

# 显示网格
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.savefig('./exp.png', dpi=600, format='png')
