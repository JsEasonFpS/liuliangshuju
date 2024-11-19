import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# 读取 CSV 文件
file_path = "D://yanjiusheng//liuliangshuju//100.csv"  # 替换为你的 CSV 文件路径
data = pd.read_csv(file_path)

# 假设数据的最后一列是标签
X = data.iloc[:, :-1].values  # 特征数据
y = data.iloc[:, -1].values    # 标签

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建主保存文件夹
output_folder = 'ceshi'
os.makedirs(output_folder, exist_ok=True)

# 按标签分类并保存每一行的一维数据为二维灰度图像
# 创建一个字典来存储每个标签的行数
label_counts = {}

for i, row in enumerate(X_scaled):
    features = row  # 标准化后的特征（不包括标签）
    label = y[i]    # 获取标签

    # 创建标签子文件夹
    label_folder = os.path.join(output_folder, str(label))
    os.makedirs(label_folder, exist_ok=True)

    # 更新当前标签的计数
    if label not in label_counts:
        label_counts[label] = 0
    else:
        label_counts[label] += 1

    # 假设每一行数据的长度为 `n`
    n = len(features)
    side_length = int(np.ceil(np.sqrt(n)))  # 计算每边的长度，确保为正方形
    image_array = np.zeros((side_length, side_length))  # 创建一个零矩阵

    # 填充矩阵
    for idx, value in enumerate(features):
        row_index = idx // side_length
        col_index = idx % side_length
        image_array[row_index, col_index] = value

    # 绘制灰度图像
    plt.figure(figsize=(5, 5))
    plt.imshow(image_array, cmap='gray', interpolation='nearest')
    #plt.title(f'Row {i} - Label: {label}')
    plt.axis('off')  # 不显示坐标轴

    # 保存图像到标签子文件夹，使用 label_counts 来确保排序
    output_file = os.path.join(label_folder, f'{label}_{label_counts[label]}.png')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭当前图像

print(f"Images saved in the '{output_folder}' folder, organized by label.")