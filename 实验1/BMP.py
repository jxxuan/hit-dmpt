# import cv2
# import sys
# import numpy as np
#
# img = cv2.imread('rgb.bmp')
# np.set_printoptions(threshold=sys.maxsize)
# print(img[200][50])    # bgr
#
import struct
import math
import matplotlib.pyplot as plt
# import os
PATH = 'image.bmp'


def getPixel(x, y):
    return pixels[y][x]


# def drawRow1(row):
#     # height = 1
#     bit = 3
#     bfType = 'BM'
#     bfSize = (width * 1 * bit + 54)
#     bfReserved1 = 0
#     bfReserved2 = 0
#     bfOffBits = 54
#     biSize = 40
#     biWidth = width
#     biHeight = 1
#     biPlanes = 1
#     biBitCount = bit * 8
#     biCompression = 0
#     biSizeImage = 0
#     biXPelsPerMeter = 3780
#     biYPelsPerMeter = 3780
#     biClrUsed = 0
#     biClrImportant = 0
#
#     with open(os.path.join(os.path.abspath('.'), 'row.bmp'), 'wb+') as file:  # reconstruct File Header
#         file.truncate(0)
#         file.write(bfType.encode('ANSI'))
#         file.write(struct.pack('i', bfSize))
#         file.write(struct.pack('h', bfReserved1))
#         file.write(struct.pack('h', bfReserved2))
#         file.write(struct.pack('i', bfOffBits))  # reconstruct bmp header
#         file.write(struct.pack('i', biSize))
#         file.write(struct.pack('i', biWidth))
#         file.write(struct.pack('i', biHeight))
#         file.write(struct.pack('h', biPlanes))
#         file.write(struct.pack('h', biBitCount))
#         file.write(struct.pack('i', biCompression))
#         file.write(struct.pack('i', biSizeImage))
#         file.write(struct.pack('i', biXPelsPerMeter))
#         file.write(struct.pack('i', biYPelsPerMeter))
#         file.write(struct.pack('i', biClrUsed))
#         file.write(struct.pack('i', biClrImportant))
#         # reconstruct pixels
#         for x in range(width):
#             file.write(struct.pack('B', getPixel(x, row)[2]))
#             file.write(struct.pack('B', getPixel(x, row)[1]))
#             file.write(struct.pack('B', getPixel(x, row)[0]))
#         os.startfile(file.name)
#
#
# def drawCol(col):
#     bit = 3
#     bfType = 'BM'
#     bfSize = (4 * height * bit + 54)
#     bfReserved1 = 0
#     bfReserved2 = 0
#     bfOffBits = 54
#     biSize = 40
#     biWidth = 1
#     biHeight = height
#     biPlanes = 1
#     biBitCount = bit * 8
#     biCompression = 0
#     biSizeImage = 0
#     biXPelsPerMeter = 3780
#     biYPelsPerMeter = 3780
#     biClrUsed = 0
#     biClrImportant = 0
#
#     with open(os.path.join(os.path.abspath('.'), 'col.bmp'), 'wb+') as file:  # reconstruct File Header
#         file.truncate(0)
#         file.write(bfType.encode('ANSI'))
#         file.write(struct.pack('i', bfSize))
#         file.write(struct.pack('h', bfReserved1))
#         file.write(struct.pack('h', bfReserved2))
#         file.write(struct.pack('i', bfOffBits))  # reconstruct bmp header
#         file.write(struct.pack('i', biSize))
#         file.write(struct.pack('i', biWidth))
#         file.write(struct.pack('i', biHeight))
#         file.write(struct.pack('h', biPlanes))
#         file.write(struct.pack('h', biBitCount))
#         file.write(struct.pack('i', biCompression))
#         file.write(struct.pack('i', biSizeImage))
#         file.write(struct.pack('i', biXPelsPerMeter))
#         file.write(struct.pack('i', biYPelsPerMeter))
#         file.write(struct.pack('i', biClrUsed))
#         file.write(struct.pack('i', biClrImportant))
#         # reconstruct pixels
#         for y in range(height):
#             file.write(struct.pack('B', getPixel(col, y)[2]))
#             file.write(struct.pack('B', getPixel(col, y)[1]))
#             file.write(struct.pack('B', getPixel(col, y)[0]))
#             file.write(b'\x00')
#         os.startfile(file.name)

def drawRow(row):
    x = range(width)
    ry, gy, by = [], [], []
    for i in range(width):
        ry.append(getPixel(i, row)[0])
        gy.append(getPixel(i, row)[1])
        by.append(getPixel(i, row)[2])
    plt.plot(x, ry, color='r')
    plt.plot(x, gy, color='g')
    plt.plot(x, by, color='b')
    plt.title("Draw Row")
    plt.show()


def drawCol(col):
    x = range(height)
    ry, gy, by = [], [], []
    for i in range(height):
        ry.append(getPixel(col, i)[0])
        gy.append(getPixel(col, i)[1])
        by.append(getPixel(col, i)[2])
    plt.plot(x, ry, color='r')
    plt.plot(x, gy, color='g')
    plt.plot(x, by, color='b')
    plt.title("Draw Col")
    plt.show()


def getHist():
    plt.subplot(3, 1, 1)
    plt.hist(rc, bins=256, color='r')
    plt.title("R")

    plt.subplot(3, 1, 2)
    plt.hist(gc, bins=256, color='g')
    plt.title("G")

    plt.subplot(3, 1, 3)
    plt.hist(bc, bins=256, color='b')
    plt.title("B")
    plt.subplots_adjust(hspace=0.7)
    plt.suptitle("Image Hist")
    plt.show()


def getEntropy():
    p = [0.0] * 256
    e = 0
    for i in range(256):
        p[i] = rc.count(i) / len(rc)
    for i in range(256):
        e = e - p[i] * math.log(p[i], 2)
    print(f"Red Entropy:{e}")
    for i in range(256):
        p[i] = gc.count(i) / len(rc)
    for i in range(256):
        e = e - p[i] * math.log(p[i], 2)
    print(f"Green Entropy:{e}")
    for i in range(256):
        p[i] = bc.count(i) / len(rc)
    for i in range(256):
        e = e - p[i] * math.log(p[i], 2)
    print(f"Blue Entropy:{e}")


with open(PATH, 'rb') as f:
    # 读取文件头
    file_header = f.read(14)

    # 读取位图信息头
    bitmap_header = f.read(40)

    # 获取图像的宽度和高度
    width = struct.unpack('<i', bitmap_header[4:8])[0]
    height = struct.unpack('<i', bitmap_header[8:12])[0]

    # 获取像素数据
    f.seek(struct.unpack('<i', file_header[10:14])[0])
    pixels = []
    for h in range(height):
        row1 = []
        for w in range(width):
            b, g, r = struct.unpack('BBB', f.read(3))
            row1.append((r, g, b))
        pixels.append(row1)

    rc, gc, bc = [], [], []
    for row2 in pixels:
        for pixel in row2:
            rc.append(pixel[0])
            gc.append(pixel[1])
            bc.append(pixel[2])

    drawRow(40)
    drawCol(40)
    getHist()
    getEntropy()
