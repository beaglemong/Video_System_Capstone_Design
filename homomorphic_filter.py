import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('12345.jfif', 0)
plt.imshow(image, cmap='gray'), plt.axis('off')
plt.show()
M, N = image.shape
P, Q = 2*M, 2*N
# Zero padding
padded_image = np.zeros((P, Q))
padded_image[:M, :N] = image
padded_image_new = np.zeros((P, Q))
# Centering
for x in range(P):
    for y in range(Q):
        padded_image_new[x, y] = padded_image[x, y] * ((-1) ** (x + y))

dft2d = np.fft.fft2(padded_image_new)
dft2d_ = np.log(np.abs(dft2d))
# Homomorphic filtering 구현
def HMMF(image, cutoff, rh, rl):
    M, N = image.shape
    H, D = np.zeros((M, N)), np.zeros((M, N))

    U0 = int(M/2)
    V0 = int(N/2)
    D0 = cutoff

    A, B = rh, rl

    for u in range(M):
        for v in range(N):
            u2 = np.power(u, 2)
            v2 = np.power(v, 2)
            D[u, v] = np.sqrt(u2 + v2)

    for u in range(M):
        for v in range(N):
            u_ = np.abs(u - U0)
            v_ = np.abs(v - V0)
            H[u, v] = (A - B) * (1 - np.exp(-D[u_, v_] ** 2 / (2 * (D0 ** 2)))) + B
# Gaussian High Pass Filter에 r_H, r_L 의 차를 곱하고 r_L을 더해 i(x, y)을 약화시키고 r(x, y)를 강화시켜 image details 강화
    return H


hmmf = HMMF(dft2d, cutoff=30, rh=1.25, rl=0.75)
plt.imshow(hmmf, cmap='gray'), plt.axis('off')
plt.show()

G = np.multiply(dft2d, hmmf)
dft2d_ = np.log(np.abs(G))
plt.imshow(dft2d_.real, cmap='gray'), plt.axis('off')
plt.show()
# Inverse Fast Fourier Transform
idft2d = np.fft.ifft2(G)

# De-centering
for x in range(P):
    for y in range(Q):
        idft2d[x, y] = idft2d[x, y] * ((-1) ** (x + y))
# Remove zero-padding
idft2d = idft2d[:M, :N]
plt.imshow(idft2d.real, cmap='gray'), plt.axis('off')
plt.show()
