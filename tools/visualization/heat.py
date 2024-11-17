import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 20 Greyc
# data = {
#     'Models': ['Asterix', 'Duck', 'Green_d', 'Red_h', 'Ball', 'Cable', 'Dragon', 'GM', 'Horse', 'Jaguar', 'LD',
#               'Mario', 'Car', '4arm', 'Rabbit', 'Statue'],
#     'Ours': [30.95, 27.21, 31.28, 29.10, 29.07, 28.03, 32.13, 25.40, 29.93, 31.88, 31.05, 27.67, 30.04, 30.85, 28.54,
#              30.82],
#     'SGW': [28.21, 27.05, 30.13, 29.07, 26.20, 25.92, 29.59, 28.82, 25.91, 29.28, 29.94, 28.72, 29.19, 29.72, 28.63,
#             29.19],
#     'Tikhonov': [27.98, 26.97, 30.13, 28.97, 26.38, 26.51, 28.95, 28.51, 26.37, 28.79, 29.21, 28.31, 28.61, 29.22,
#                  28.41, 28.95],
#     'TV': [23.67, 23.57, 23.34, 23.46, 24.31, 23.52, 23.39, 23.67, 23.36, 23.67, 23.45, 24.29, 23.61, 23.32, 24.07,
#            23.32],
#     'GLR': [28.1, 26.89, 30.30, 27.85, None, None, None, None, None, None, None, None, None, None, None, None],
#     'GTV': [27.56, 26.35, 30.34, 28.57, None, None, None, None, None, None, None, None, None, None, None, None],
#     'MBF': [29.31, None, 30.12, 28.13, 28.93, None, None, None, None, None, None, None, None, None, None, None]}

# 30 Greyc
# data = {
#     'Models': ['Asterix', 'Duck', 'Green_d', 'Red_h', 'Ball', 'Cable', 'Dragon', 'GM', 'Horse', 'Jaguar', 'LD',
#                'Mario', 'Car', '4arm', 'Rabbit', 'Statue'],
#     'Ours': [30.14, 25.51, 30.52, 27.80, 28.72, 26.71, 30.79, 24.48, 28.42, 30.71, 29.96, 27.00, 29.21, 29.18, 27.02,
#              29.32, ],
#     'SGW': [27.18, 25.90, 29.01, 27.06, 25.25, 24.61, 28.14, 26.96, 24.59, 27.13, 28.83, 26.75, 27.73, 28.43, 26.87,
#             27.99, ],
#     'Tikhonov': [26.42, 25.36, 28.63, 26.88, 24.14, 24.81, 27.31, 26.72, 24.76, 26.99, 27.63, 26.16, 26.78, 27.73,
#                  26.33, 27.23, ],
#     'TV': [20.81, 20.74, 20.39, 20.64, 21.61, 20.61, 20.43, 20.79, 20.37, 20.81, 20.53, 21.43, 20.78, 20.42, 21.27,
#            20.33, ],
#     'GLR': [26.12, 25.68, 28.52, 25.66, None, None, None, None, None, None, None, None, None, None, None, None],
#     'GTV': [27.05, 25.04, 29.62, 27.33, None, None, None, None, None, None, None, None, None, None, None, None],}

# # 20 8i
# data = {
#     'Models': ['RedAndBlack', 'LongDress', 'Loot', 'Soldier'],
#     'Ours': [30.08, 29.73, 35.35, 34.63, ],
#     'SGW': [29.82, 27.01, 31.15, 29.95, ],
#     'IL-GLF': [31.30, 29.41, 32.37, 31.63, ],
#     'GLR': [31.18, 28.86, 32.36, 31.29, ],
#     '3DPBS': [32.99, 29.90, 33.85, 32.50, ],
# }

# 30 8i
data = {
    'Models': ['RedAndBlack', 'LongDress', 'Loot', 'Soldier'],
    'Ours': [29.22, 28.27, 33.57, 33.10, ],
    'SGW': [28.96, 26.76, 30.17, 29.71, ],
    'IL-GLF': [28.73, 27.54, 29.35, 28.87, ],
    'GLR': [28.79, 27.27, 29.50, 28.84, ],
    '3DPBS': [30.66, 28.33, 32.51, 30.48, ],
}

# 转换为 DataFrame
df = pd.DataFrame(data)
df.set_index('Models', inplace=True)

# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap='Reds', cbar_kws={'label': 'PSNR'}, mask=df.isnull(), linewidths=0.5, fmt=".2f")

plt.xlabel('Algorithms')
plt.ylabel('Models')
plt.savefig('./heat.png', dpi=600, format='png')
