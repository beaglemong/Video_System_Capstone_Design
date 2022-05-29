import cv2
import numpy as np

def GaussianFilter(img):
  gaussian_kernel = np.array([[1,2,1], [2,4,2], [1,2,1]]) / 16
  
  H, W = img.shape
  pad_img = np.zeros((H+ 2, W + 2))
  pad_img[1:H+1, 1:W+1] = img
  
  filtered_img = np.zeros(img.shape)
  
  for i in range(H):
    for j in range(W):
      filtered_img[i, j] = np.sum(pad_img[i:i+3, j:j+3] * gaussian_kernel)

  return filtered_img
