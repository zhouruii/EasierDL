import matplotlib.pyplot as plt
import numpy as np

# 示例数据：各个算法的 PSNR 值
# algorithms = ['GLR', 'SGW', 'IL-GLF', '3DPBS', 'FGBD', 'Ours']
# # psnr_values = [30.92,29.48,31.17,32.31,31.77,33.49,]  # 这些值可以是单张图片的 PSNR 值或平均值
# psnr_values = [52.6, 73.3, 49.7, 38.2, 43.3, 29.1]  # 这些值可以是单张图片的 PSNR 值或平均值

algorithms = ['Ablation-V1', 'Ablation-V2', 'PCCDNet']
psnr_values = [28.14, 28.68, 29.62]

# 绘制直方图
plt.figure(figsize=(8, 5))
bars = plt.bar(algorithms, psnr_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], width=0.3)

# 添加每个条形的文本标记
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom', ha='center')

# 设置坐标轴标签和标题
plt.ylabel('PSNR')

# 显示图形
plt.tight_layout()
# plt.xticks(rotation=15)  # 调整 x 轴标签角度
plt.savefig('./bar.png', dpi=600, format='png')
