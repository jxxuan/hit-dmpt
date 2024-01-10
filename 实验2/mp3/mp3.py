import subprocess
import time
import wave
import os

# 生成文件名
pcm_filename = 'input.pcm'
mp3_filename = 'output.mp3'
decoded_pcm_filename = 'decoded.pcm'

# 删除已有文件
filenames = [pcm_filename, mp3_filename, decoded_pcm_filename]
for filename in filenames:
    if os.path.exists(filename):
        os.remove(filename)

# 打开WAV文件
with wave.open('audio.wav', 'rb') as wav_file:
    # 获取采样率、采样宽度和声道数
    frameRate = wav_file.getframerate()
    sample_width = wav_file.getsampwidth()
    channels = wav_file.getnchannels()
    # 读取PCM音频数据
    pcm_data = wav_file.readframes(-1)

# 将PCM数据写入临时文件
with open(pcm_filename, 'wb') as pcm_file:
    pcm_file.write(pcm_data)

# 将PCM文件进行MP3编码
encode_command = f"ffmpeg -f s{8*sample_width}le -ar {frameRate} -ac {channels} -i {pcm_filename} {mp3_filename}"
start_time = time.time()
subprocess.call(encode_command, shell=True)
encoding_time = time.time() - start_time

# 统计压缩前后文件大小
pcm_filesize = len(pcm_data)
mp3_filesize = len(open(mp3_filename, 'rb').read())
# 计算压缩倍数
compression_ratio = mp3_filesize / pcm_filesize


# 将MP3文件进行解码
decode_command = f"ffmpeg -i {mp3_filename} -f s{8*sample_width}le -acodec pcm_s{8*sample_width}le -ar {frameRate} -ac {channels} {decoded_pcm_filename}"
start_time = time.time()
subprocess.call(decode_command, shell=True)
decoding_time = time.time() - start_time

# 读取解码后的PCM数据
with open(decoded_pcm_filename, 'rb') as pcm_file:
    decoded_pcm_data = pcm_file.read()

# 打印统计信息
print("压缩时间：", encoding_time, "秒")
print("压缩前文件大小：", pcm_filesize, "字节")
print("压缩后文件大小：", mp3_filesize, "字节")
print("压缩倍数：{:.5f}%".format(compression_ratio * 100))
print(f"解压缩时间：", decoding_time, "秒")
