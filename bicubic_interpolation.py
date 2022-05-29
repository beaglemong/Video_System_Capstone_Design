import cv2
import numpy as np

image = cv2.imread('image.jpg')

H, W, C = image.shape
scaling = 2

#zero-padding
pad_img = np.zeros((H, W+3, C))
pad_img[:, 1:W+1, :] = image


image_lerp1 = np.zeros((H, W*scaling, C))


for i in range(H):
    for j in range(W):
        image_lerp1[i, 2*j, :] = image[i, j, :]
cv2.imshow('Step 1-1', image_lerp1.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()

#bicubic interpolation
for i in range(H):
    for j in range(W-1):
        p0, p1, p2, p3 = pad_img[i, j], pad_img[i, j+1], pad_img[i, j+2], pad_img[i, j+3]
        image_lerp1[i, 2 * j + 1, :] = ((-1 / 2) * p0 + (3 / 2) * p1 + (-3 / 2) * p2 + (1 / 2) * p3) * (0.5 ** 3) + \
                                       (p0 + (-5 / 2) * p1 + 2 * p2 + (-1 / 2) * p3) * (0.5 ** 2) + \
                                       ((-1 / 2) * p0 + (1 / 2) * p2) * 0.5 + p1
cv2.imshow('Step 1-2', image_lerp1.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()


#zero-padding
pad_img_lerp = np.zeros((H + 3, W*scaling, C))
pad_img_lerp[1:H+1, :, :] = image_lerp1


interpolated_image = np.zeros((H*scaling, W*scaling, C))
for i in range(H):
    for j in range(W*scaling):
        interpolated_image[2*i, j, :] = image_lerp1[i, j, :]
cv2.imshow('Step 2-1', interpolated_image.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()


for i in range(H):
    for j in range(W*scaling-1):
        p0, p1, p2, p3 = pad_img_lerp[i, j], pad_img_lerp[i + 1, j], pad_img_lerp[i + 2, j], pad_img_lerp[i + 3, j]
        interpolated_image[2 * i + 1, j, :] = ((-1 / 2) * p0 + (3 / 2) * p1 + (-3 / 2) * p2 + (1 / 2) * p3) * (0.5 ** 3) + \
                                       (p0 + (-5 / 2) * p1 + 2 * p2 + (-1 / 2) * p3) * (0.5 ** 2) + \
                                       ((-1 / 2) * p0 + (1 / 2) * p2) * 0.5 + p1



interpolated_image = np.where(interpolated_image > 255, 255, interpolated_image)
interpolated_image = np.where(interpolated_image < 0, 0, interpolated_image)


cv2.imshow('Original image', image)
cv2.imshow('Resized image', interpolated_image.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()
