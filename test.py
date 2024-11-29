import numpy as np
import matplotlib.pyplot as plt
import os

# 生成单位矩阵
unit_matrix = np.eye(5)  # 创建一个5x5的单位矩阵

# 将单位矩阵保存为图像文件
plt.imsave('unit_matrix.png', unit_matrix, cmap='gray')

# 获取当前工作目录
current_path = os.getcwd()

# 报告总结
print(f"图像被写入的位置（路径）：{current_path}")
print(f"图像的类型（扩展名）：.png")