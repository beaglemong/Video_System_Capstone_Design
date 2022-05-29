import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_img():
    img = np.zeros((7, 15))
    for i in range(7):
        for j in range(15):
            if i == 1 or i == 5:
                if j >= 1 and j <= 4:
                    img[i, j] = 1
                if j >= 10 and j <= 13:
                    img[i, j] = 1
            if i == 2 or i == 4:
                if j >=1 and j <= 13:
                    img[i, j] = 1
            if i == 3:
                if j >=1 and j <= 13:
                    img[i, j] = 1
                img[i, 5] = 0
                img[i, 9] = 0
    img = img.astype(np.uint8)
    return img

def get_img_reverse():
    img = np.ones((7, 15))
    for i in range(7):
        for j in range(15):
            if i == 1 or i == 5:
                if j >= 1 and j <= 4:
                    img[i, j] = 0
                if j >= 10 and j <= 13:
                    img[i, j] = 0
            if i == 2 or i == 4:
                if j >=1 and j <= 13:
                    img[i, j] = 0
            if i == 3:
                if j >=1 and j <= 13:
                    img[i, j] = 0
                img[i, 5] = 1
                img[i, 9] = 1
    img = img.astype(np.uint8)
    return img  # img 여집합 생성

def erosion(boundary=None, kernel=None):
    boundary = boundary * kernel
    if(np.min(boundary) == 0):
        return 0
    else:
        return 255   # erosion


def morphology(img, method, k_size):
    h, w = img.shape
    pad = k_size // 2
    pad_img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
    result = img.copy()
    B1 = np.array([[1, 1, 1],
                  [1, -1, 1],
                  [1, 1, 1]], dtype="int")
    B2 = np.array([[-1, -1, -1],
                  [-1, 1, -1],
                  [-1, -1, -1]], dtype="int")

    if method == 1 or method == 2:
        for i in range(h):
            for j in range(w):
                if method == 1:
                    result[i, j] = erosion(pad_img[i:i+k_size, j:j+k_size], B1)
                elif method == 2:
                    result[i, j] = erosion(pad_img[i:i+k_size, j:j+k_size], B2)

    return result  # 각각의 image에 B1(method=1)과 B2(method=2)의 erosion 실행


def intersaction(img, img1):
    H, W = img.shape
    result_img = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            if img[i, j] == 255 or img1[i, j] == 255:
                result_img[i, j] = 0
            else:
                result_img[i, j] = 255
    return result_img  # 교집합 생성
###########################
# Homework : implement this function #
###########################
def hit_or_miss():
    img = get_img()
    img1 = get_img_reverse()
    # plt.imshow(img, cmap="gray"), plt.show()
    # plt.imshow(img1, cmap="gray"), plt.show()

    B1 = np.array([[1, 1, 1],
                   [1, -1, 1],
                   [1, 1, 1]], dtype="int")
    B2 = np.array([[-1, -1, -1],
                   [-1, 1, -1],
                   [-1, -1, -1]], dtype="int")
    # plt.imshow(B1, cmap="gray"), plt.show()
    # plt.imshow(B2, cmap="gray"), plt.show()
    kernel_size = 3
    ERO1 = morphology(img, method=1, k_size=kernel_size)
    ERO2 = morphology(img1, method=2, k_size=kernel_size)

    # plt.imshow(ERO1, cmap="gray"), plt.show()
    # plt.imshow(ERO2, cmap="gray"), plt.show()

    HMT_result = intersaction(ERO1, ERO2)
    plt.imshow(HMT_result, cmap="gray"), plt.show()  # hit_or_miss 함수 구현
###########################

if __name__ == "__main__":    
    img = get_img()
    
    B = np.array([[1, 1, 1],
                  [1, -1, 1],
                  [1, 1, 1]], dtype="int")
    
    #Built-in function
    cv_result = cv2.morphologyEx(img, cv2.MORPH_HITMISS, B)  # 내장함수를 통한 hit-or-miss
    #Implementation
    plt.imshow(cv_result, cmap="gray"), plt.show()
    hit_or_miss()  # 구현한 hit-or-miss
