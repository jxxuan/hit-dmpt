import cv2
import numpy as np


def PermutationFun(inputImage, blockWidth, blockHeight, seed):
    # 读取输入图像
    img = cv2.imread(inputImage)

    # 获取图像的宽度和高度
    height, width = img.shape[:2]

    # 计算图像中块的数量
    num_blocks_x = int(np.ceil(width / blockWidth))
    num_blocks_y = int(np.ceil(height / blockHeight))
    blocks = []

    # 遍历每个块并置乱其位置
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            # 计算当前块的左上角和右下角坐标
            x1 = x * blockWidth
            y1 = y * blockHeight
            x2 = min(x1 + blockWidth, width)
            y2 = min(y1 + blockHeight, height)
            # 提取当前块的像素值
            blocks.append(img[y1:y2, x1:x2].copy())
    np.random.seed(seed)
    np.random.shuffle(blocks)

    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            # 计算当前块的左上角和右下角坐标
            x1 = x * blockWidth
            y1 = y * blockHeight
            x2 = min(x1 + blockWidth, width)
            y2 = min(y1 + blockHeight, height)
            # 将打乱后的像素值写回到图像中
            img[y1:y2, x1:x2] = blocks.pop()

    # 显示置乱后的图像
    cv2.namedWindow('Permutation', cv2.WINDOW_NORMAL)
    cv2.imshow('Permutation', img)
    cv2.waitKey(0)


def CropFun(inputImage, x1, y1, x2, y2, outputImage):
    # 读取输入图像
    img = cv2.imread(inputImage)

    # 截取图像的一个区域
    roi = img[y1:y2, x1:x2]

    # 将截取的区域保存为另一幅图像
    cv2.imwrite(outputImage, roi)


PermutationFun('image.bmp', 64, 64, 8)
CropFun('image.bmp', 100, 100, 200, 200, 'crop.bmp')
