import cv2
import numpy as np

bgr = cv2.imread("lena.jpg")
bgr_norm = bgr / 255.
b, g, r = cv2.split(bgr)
hsi = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
h, s, i = cv2.split(hsi)

H, W, C = bgr.shape
pad_bgr = np.zeros([H + 2, W + 2, C])  # zero padding
pad_bgr[1: H + 1, 1: W + 1, 0] = bgr[:, :, 0]
pad_bgr[1: H + 1, 1: W + 1, 1] = bgr[:, :, 1]
pad_bgr[1: H + 1, 1: W + 1, 2] = bgr[:, :, 2]

filtered_bgr = np.zeros(bgr.shape)
mask = np.ones((3, 3), np.float32) / 9  # filtering mask

for i in range(H):
    for j in range(W):
        filtered_bgr[i, j, 0] = np.sum(pad_bgr[i:i + 3, j:j + 3, 0] * mask)
        filtered_bgr[i, j, 1] = np.sum(pad_bgr[i:i + 3, j:j + 3, 1] * mask)
        filtered_bgr[i, j, 2] = np.sum(pad_bgr[i:i + 3, j:j + 3, 2] * mask)  # RGB에서의 filtering


X, Y, Z = hsi.shape
pad_hsi = np.zeros([X + 2, Y + 2, Z])
pad_hsi[1: H + 1, 1: W + 1, 1] = hsi[:, :, 1]
pad_hsi[1: H + 1, 1: W + 1, 2] = hsi[:, :, 2]
pad_hsi[1: H + 1, 1: W + 1, 0] = hsi[:, :, 0]

filtered_hsi = np.zeros(hsi.shape)
filtered_hsi[:, :, 0] = hsi[:, :, 0]
filtered_hsi[:, :, 1] = hsi[:, :, 1]

for i in range(X):
    for j in range(Y):
        filtered_hsi[i, j, 2] = np.sum(pad_hsi[i: i + 3, j: j + 3, 2] * mask)  # HSI intensity 영역에서 filtering


convert_to_rgb = cv2.cvtColor(filtered_hsi.astype(np.uint8), cv2.COLOR_HSV2BGR)

difference = filtered_bgr - convert_to_rgb


cv2.imshow('Before filtering BGR', bgr)
cv2.imshow('After filtering BGR', filtered_bgr.astype(np.uint8))
cv2.imshow('Before filtering HSI', hsi)
cv2.imshow('After filtering HSI', filtered_hsi.astype(np.uint8))
cv2.imshow('After filtering HSI -> BGR', convert_to_rgb)
cv2.imshow('Difference', difference.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()
