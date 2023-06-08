import numpy as np
import matplotlib.pyplot as plt
import cv2

# 读取图像并进行傅里叶变换
img = cv2.imread('njust.png', 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# 构建高斯滤波器
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)
radius = 30
mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (crow, ccol), radius, 1, -1)

# 进行频域滤波，并进行傅里叶反变换
fshift = fshift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# 显示原图像和处理后的图像
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Processed Image'), plt.xticks([]), plt.yticks([])
plt.show()

# 计算差异并显示差异图像
diff = np.abs(img - img_back)
plt.imshow(diff, cmap='gray')
plt.title('Difference Image'), plt.xticks([]), plt.yticks([])
plt.show()