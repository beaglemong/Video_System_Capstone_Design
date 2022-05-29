import cv2
import numpy as np
from matplotlib import pyplot as plt


def histo_equal(img):
    N, M = img.shape
    G = 256  # 0 ~255의 gray scale 분포
    his = np.zeros(G)  # histogram equalization 구현을 위한 array

    for i in img.ravel():
        his[i] += 1  # image 의 pixel 만큼 for loop 을 돌며 histogram array 의 값을 채움

    g_min = np.min(np.nonzero(his))  # image 의 pixel 중 개수가 가장 최소인 pixel 의 값 반환

    his_cdf = np.zeros_like(his)  # histogram array 의 크기를 가지는 0으로 채워진 array 생성
    his_cdf[0] = his[0]

    for g in range(1, G):
        his_cdf[g] = his_cdf[g - 1] + his[g]  # histogram array 의 cdf array

    his_min = his_cdf[g_min]

    T = np.round((his_cdf - his_min) / (M * N - his_min) * (G - 1))  # transformation function

    result = np.zeros_like(img)  # transformation function 을 통해 result image 값 저장 하기 위한 array

    for n in range(N):
        for m in range(M):
            result[n, m] = T[img[n, m]]
    return result  # transform 된 pixel result array 에 저장


def show_histo(img, image):  # histogram 분포를 분석 하기 위해 함수 생성
    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax0.hist(img.ravel(), 256, [0, 256])
    ax1 = fig.add_subplot(212)
    ax1.hist(image.ravel(), 256, [0, 256])
    plt.show()


img = cv2.imread('bean.tif', 0)
result = histo_equal(img)
show_histo(img, result)
cv2.imshow('original', img)
cv2.imshow('after_histogram_equalization', result.astype(np.uint8))
cv2.waitKey()

img1 = cv2.imread('bean_dark.tif', 0)
result1 = histo_equal(img1)
show_histo(img1, result1)
cv2.imshow('dark_original', img1)
cv2.imshow('after_histogram_equalization', result1.astype(np.uint8))
cv2.waitKey()

img2 = cv2.imread('bean_light.tif', 0)
result2 = histo_equal(img2)
show_histo(img2, result2)
cv2.imshow('light_original', img2)
cv2.imshow('after_histogram_equalization', result2.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()
