import numpy as np
import matplotlib.pyplot as plt

# 信号频率
f1 = 30
f2 = 35

# 采样区间和采样频率
T = 0.02
fs = 1 / T

# 数据长度
lengths = [10, 15, 30, 40, 60, 70, 100]

# 填充长度
fill_length = 512

# 生成时间序列
t = np.arange(0, fill_length, 1 / fs)

# 初始化图像
fig, axs = plt.subplots(len(lengths), 1, figsize=(8, 6))

# 逐个数据长度进行采样和填充
for i, length in enumerate(lengths):
    # 生成信号
    signal1 = np.sin(2 * np.pi * f1 * t[:length])
    signal2 = np.sin(2 * np.pi * f2 * t[:length])

    # 零填充
    signal1_padded = np.pad(signal1, (0, fill_length - length), 'constant')
    signal2_padded = np.pad(signal2, (0, fill_length - length), 'constant')

    # 绘制信号图像
    axs[i].plot(signal1_padded, label='Signal 1')
    axs[i].plot(signal2_padded, label='Signal 2')
    axs[i].set_title(f'Length: {length}')
    axs[i].legend()

plt.tight_layout()
plt.show()
