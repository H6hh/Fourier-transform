import cv2
import numpy as np

img = cv2.imread('njust.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))
phase_spectrum = np.angle(fshift)

fshift1 = np.zeros((gray.shape[0], gray.shape[1]), dtype='complex')
fshift1[magnitude_spectrum > 100] = fshift[magnitude_spectrum > 100]
f_ishift1 = np.fft.ifftshift(fshift1)
img_back1 = np.abs(np.fft.ifft2(f_ishift1))

diff = cv2.absdiff(gray, img_back1.astype(np.uint8))

cv2.imshow('Input Image', gray)
cv2.imshow('Filtered Image', img_back1.astype(np.uint8))
cv2.imshow('Difference', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()