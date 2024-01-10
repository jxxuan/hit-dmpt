from PIL import Image


# 压缩BMP图像为JPEG
def compress_bmp_to_jpeg(input_path, output_path, quality=85):
    image = Image.open(input_path)
    image.save(output_path, "JPEG", quality=quality)


# 解压缩JPEG图像为BMP
def decompress_jpeg_to_bmp(input_path, output_path):
    image = Image.open(input_path)
    image.save(output_path, "BMP")


# 文件名
bmp_path = "image.bmp"
jpeg_path = "compressed.jpeg"
decompressed_bmp_path = "decompressed.bmp"

# 压缩BMP为JPEG
compress_bmp_to_jpeg(bmp_path, jpeg_path, quality=85)

# 解压缩JPEG为BMP
decompress_jpeg_to_bmp(jpeg_path, decompressed_bmp_path)
