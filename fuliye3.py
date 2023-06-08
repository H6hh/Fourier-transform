import cv2
import numpy as np

# 读取图像文件
img = cv2.imread('njust.png')

# 判断图像是否为空
if img is None:
    print('无法读取图像文件')
else:
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 傅里叶变换计算
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    phase_spectrum = np.angle(fshift)

    # 幅度谱为常数的傅里叶反变换
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    fshift1 = np.zeros((rows, cols), dtype='complex')
    fshift1[crow - 30: crow + 30, ccol - 30: ccol + 30] = magnitude_spectrum[crow - 30: crow + 30, ccol - 30: ccol + 30] + phase_spectrum[crow - 30: crow + 30, ccol - 30: ccol + 30] * 1j
    f_ishift1 = np.fft.ifftshift(fshift1)
    img_back1 = np.abs(np.fft.ifft2(f_ishift1))

    # 显示图像
    diff = cv2.absdiff(gray, img_back1.astype(np.uint8))
    cv2.imshow('Input Image', gray)
    cv2.imshow('Filtered Image', img_back1.astype(np.uint8))
    cv2.imshow('Difference', diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()