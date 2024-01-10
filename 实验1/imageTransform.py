import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import timeit


def zigzag_scan(matrix, k):
    # 获取矩阵的行数和列数
    rows, cols = matrix.shape

    # 创建一个一维数组，用于存储Zigzag扫描后的系数
    zigzag = np.zeros(rows * cols, dtype=matrix.dtype)

    # 初始化索引变量和方向
    row, col = 0, 0
    direction = 1

    # 遍历矩阵并按照Zigzag顺序填充数组
    for i in range(k):
        zigzag[i] = matrix[row, col]

        # 根据当前位置和方向更新下一个位置
        if direction == 1:
            # 向上扫描
            if col == cols - 1:
                row += 1
                direction = -1
            elif row == 0:
                col += 1
                direction = -1
            else:
                row -= 1
                col += 1
        else:
            # 向下扫描
            if row == rows - 1:
                col += 1
                direction = 1
            elif col == 0:
                row += 1
                direction = 1
            else:
                row += 1
                col -= 1

    return zigzag


def zigzag_restore(zigzag, rows, cols):
    # 创建一个空的矩阵，用于存储还原后的系数
    matrix = np.zeros((rows, cols), dtype=zigzag.dtype)

    # 初始化索引变量和方向
    row, col = 0, 0
    direction = 1

    # 遍历一维数组并按照Zigzag顺序填充矩阵
    for i in range(rows * cols):
        matrix[row, col] = zigzag[i]

        # 根据当前位置和方向更新下一个位置
        if direction == 1:
            # 向上扫描
            if col == cols - 1:
                row += 1
                direction = -1
            elif row == 0:
                col += 1
                direction = -1
            else:
                row -= 1
                col += 1
        else:
            # 向下扫描
            if row == rows - 1:
                col += 1
                direction = 1
            elif col == 0:
                row += 1
                direction = 1
            else:
                row += 1
                col -= 1

    return matrix


def compress_image(image, k):
    height, width = image.shape
    compressed_image = np.zeros((height, width), dtype=np.float32)

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            # 图像分块
            block = np.float32(image[i:i + 8, j:j + 8])
            # DCT变换
            dct_block = cv2.dct(block)
            # 做Zigzag扫描，保留左上角k个数据，进行逆变换
            compressed_block = zigzag_restore(zigzag_scan(dct_block, k), 8, 8)
            idct_block = cv2.idct(compressed_block)
            # 得到压缩后的图像
            compressed_image[i:i + 8, j:j + 8] = idct_block

    return compressed_image


def fft():
    np.fft.fft2(image)


# 读取图像
image = cv2.imread('image.bmp', cv2.IMREAD_GRAYSCALE)

# 二维 DFT
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('DFT Magnitude'), plt.xticks([]), plt.yticks([])
plt.show()

# FFT 变换速度
fft_time = timeit.timeit(fft, number=1)
print("FFT变换时间：", fft_time, "秒")


# 二维 DCT
k = 20  # 保留的系数个数
compressed_image = compress_image(image, k)

plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(compressed_image, cmap='gray')
plt.title(f"DCT Compress(k={k})"), plt.xticks([]), plt.yticks([])
plt.show()


# DWT 变换
wavelet = 'haar'  # 小波类型
coeffs = pywt.dwt2(image, wavelet)
# DWT 系数
cA, (cH, cV, cD) = coeffs
# 恢复图像
restored_image = pywt.idwt2((cA, (cH, cV, cD)), wavelet)

plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(restored_image, cmap='gray')
plt.title("DWT Result"), plt.xticks([]), plt.yticks([])
plt.show()

# 计算 PSNR 和 SSIM，同时指定 data_range
psnr = peak_signal_noise_ratio(image, compressed_image, data_range=255)
ssim = structural_similarity(image, compressed_image, data_range=255)

print('PSNR:', psnr)
print('SSIM:', ssim)
