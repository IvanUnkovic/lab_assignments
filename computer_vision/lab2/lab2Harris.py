from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def harrisov_odziv(velicina_prozora, k, prag, topk):

    sobel_operator_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_operator_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradijent_x = convolve(img, sobel_operator_x)
    gradijent_y = convolve(img, sobel_operator_y)

    Ix_kv = gradijent_x ** 2
    Iy_kv = gradijent_y ** 2
    Ixy = gradijent_x * gradijent_y

    plt.subplot(1, 2, 1)
    plt.imshow(gradijent_x, cmap='gray')
    plt.title('Gradijent po X osi')
    plt.subplot(1, 2, 2)
    plt.imshow(gradijent_y, cmap='gray')
    plt.title('Gradijent po Y osi')
    plt.show()

    matrica_lokalnog_susjedstva = np.ones((velicina_prozora, velicina_prozora))
    Ix_kv_suma = convolve(Ix_kv, matrica_lokalnog_susjedstva)
    Iy_kv_suma = convolve(Iy_kv, matrica_lokalnog_susjedstva)
    Ixy_suma = convolve(Ixy, matrica_lokalnog_susjedstva)

    r = Ix_kv_suma * Iy_kv_suma - Ixy_suma**2 - k * (Ix_kv_suma + Iy_kv_suma)**2

    plt.imshow(r, cmap='gray')
    plt.title("Harrisov odziv")
    plt.show()

    r[r < prag] = 0
    for i in range(1, r.shape[0]-1):
        for j in range(1-r.shape[1]-1):
            if prag < r[i,j]:
                susjedi = [r[i-1,j-1], r[i-1,j], r[i-1, j+1], r[i, j-1], r[i, j+1], r[i+1,j-1], r[i+1, j], r[i+1, j+1]]
                if r[i,j] < np.max(susjedi):
                    r[i,j] = 0

    plt.imshow(r, cmap='gray')
    plt.title("Harrisov odziv s potiskivanjem")
    plt.show()

    return r


img = np.array(Image.open("../fer_logo.jpg").convert('L'))
print("Dimenzije slike fer-loga: {}".format(img.shape))
min_I = np.min(img)
max_I = np.max(img)
print("Najmanji intenzitet: {}".format(min_I))
print("Najveći intenzitet: {}".format(max_I))
gornji_lijevi_isjecak = img[:10, :10] 
print("Intenziteti gornjeg lijevog isječka od 10x10 piksela:\n {}".format(gornji_lijevi_isjecak))
print("Slika je tipa: {}".format(img.dtype))
img = img.astype(np.float32)
print("Slika je sada tipa: {}".format(img.dtype))

sigmas = [1,2,5,10]
for i, sigma in enumerate(sigmas):
    smooth_img = gaussian_filter(img, sigma=sigma)
    plt.subplot(2, 2, i+1)
    plt.imshow(smooth_img, cmap="gray")
    plt.title("Sigma = {}".format(sigma))
plt.show()


kutevi = harrisov_odziv(velicina_prozora=3, k=0.04, prag=0.1, topk=100)



