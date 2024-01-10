ffmpeg -i image.bmp image.ppm
ffmpeg -i image.ppm -pix_fmt monob image.pbm
pbmtojbg image.pbm image.jbg