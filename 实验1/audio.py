import wave
import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.fft as fft

PATH = 'audio.wav'


# 打开WAV文件
with wave.open(PATH, 'rb') as wav_file:
    # 获取采样率和采样宽度
    frameRate = wav_file.getframerate()
    sample_width = wav_file.getsampwidth()

    # 读取PCM音频数据
    pcm_data = wav_file.readframes(-1)

# 将PCM数据转换为NumPy数组
pcm_array = np.frombuffer(pcm_data, dtype=np.int16)

# 将左右声道分别存储到两个数组中
left_channel = pcm_array[::2]
right_channel = pcm_array[1::2]

# 计算时间轴
time_axis = np.arange(len(left_channel)) / frameRate

# 绘制波形图
plt.subplot(2, 1, 1)
plt.plot(time_axis, left_channel)
plt.title("L")
plt.subplot(2, 1, 2)
plt.plot(time_axis, right_channel)
plt.title("R")
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# 定义窗口长度
window_size = 1024

# 计算窗口数量
num_windows = len(pcm_array) // window_size

# 初始化频谱图
spectrogram = np.zeros((10, num_windows))
# 进行分窗和DFT计算
for i in range(num_windows):
    # 计算当前窗口的起始和结束索引
    start = i * window_size
    end = start + window_size
    # 取当前窗口的音频数据
    window = pcm_array[start:end]
    # 对当前窗口的音频数据进行DFT计算
    spectrum = np.abs(np.fft.fft(window)[:window_size // 2 + 1])[:10]
    # spectrum = np.log10(spectrum + 1e-10)
    # 将当前窗口的频谱数据添加到频谱图中
    spectrogram[:, i] = spectrum
# 创建频率轴和时间轴
freq_axis = np.fft.fftfreq(window_size, d=1 / frameRate)[:window_size // 2 + 1]
time_axis = np.arange(num_windows) * window_size / frameRate / 2
# 绘制频谱图
plt.figure()
plt.imshow(spectrogram, aspect='auto', origin='lower',
           extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]], cmap='inferno')
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')
plt.show()


# 初始化DCT结果
dct_result = np.zeros((window_size, num_windows))
for i in range(num_windows):
    # 计算当前窗口的起始和结束索引
    start = i * window_size
    end = start + window_size
    # 取当前窗口的音频数据
    window = pcm_array[start:end]
    # 进行DCT变换
    dct_result[:, i] = fft.dct(window, type=2, norm="ortho")
# 绘制DCT变换后的图形
plt.imshow(dct_result, aspect='auto', origin='lower')
plt.colorbar()
plt.xlabel('Window')
plt.ylabel('DCT Coefficient')
plt.title('DCT Transform')
plt.show()


# 归一化音频数据
audio_data = pcm_array / np.max(np.abs(pcm_array))

# 定义小波类型和层数
wavelet_type = 'db4'  # 小波类型，例如 Daubechies 4 (db4)
level = 5  # DWT 的层数

# 进行离散小波变换 (DWT)
coeffs = pywt.wavedec(audio_data, wavelet_type, level=level)

# 提取近似系数和细节系数
approximation = coeffs[0]  # 近似系数
details = coeffs[1:]  # 细节系数

# 绘制DWT系数
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为Microsoft YaHei
plt.subplot(level + 1, 1, 1)
plt.plot(approximation)
plt.title('近似系数')
for i, detail in enumerate(details):
    plt.subplot(level + 1, 1, i + 2)
    plt.plot(detail)
    plt.title(f'细节系数 - 第 {i + 1} 层')
plt.tight_layout()
plt.show()
