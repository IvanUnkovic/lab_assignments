from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import math

def normalizacija_magnitude(magnituda):
    return (magnituda / np.max(magnituda)) * 255

def jaki_rubovi(img, max_val, min_val):
    for i in range(1, img.shape[0]-1):
        for j in range(1-img.shape[1]-1):
            if img[i,j]<=min_val:
                img[i,j] = 0
            if min_val <= img[i,j] <= max_val:
                susjedi = [img[i-1,j-1], img[i-1,j], img[i-1, j+1], img[i, j-1], img[i, j+1], img[i+1,j-1], img[i+1, j], img[i+1, j+1]]
                if not np.any(susjedi > max_val):
                    img[i,j] = 0
    return img

img = np.array(Image.open("../house.jpg").convert('L'))
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

new_img = gaussian_filter(img, sigma=3)

sobel_operator_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_operator_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

gradijent_x = convolve(new_img, sobel_operator_x)
gradijent_y = convolve(new_img, sobel_operator_y)

magnituda = np.sqrt(gradijent_x**2 + gradijent_y**2)
kut = np.arctan(gradijent_y/gradijent_x)

norm_m = normalizacija_magnitude(magnituda)

plt.imshow(norm_m, cmap="gray")
plt.show()

redovi, stupci = magnituda.shape


for i in range(1, redovi - 1):
    for j in range(1, stupci - 1):
        trenutni_pixel = norm_m[i,j]
        trenutni_kut = math.degrees(kut[i, j])
        if -22.5 <= trenutni_kut <= 22.5 or 157.5 <= trenutni_kut <= 180 or -180 <= trenutni_kut <= -157.5:
            susjedi = [norm_m[i, j-1], norm_m[i, j+1]]
        elif 67.5 <= trenutni_kut <= 112.5 or -112.5 <= trenutni_kut <= -67.5:
            susjedi = [norm_m[i-1, j], norm_m[i+1, j]]
        elif 22.5 <= trenutni_kut <= 67.5 or -157.5 <= trenutni_kut <= -112.5:
            susjedi = [norm_m[i - 1, j + 1], norm_m[i + 1, j - 1]]
        elif 112.5 <= trenutni_kut <= 157.5 or -67.5 <= trenutni_kut <= -22.5:
            susjedi = [norm_m[i - 1, j - 1], norm_m[i + 1, j + 1]]
        if trenutni_pixel < max(susjedi):
            norm_m[i, j] = 0


plt.imshow(norm_m, cmap="gray")
plt.show()   

jaki_rubovi_img = jaki_rubovi(norm_m, 90, 10)
plt.imshow(jaki_rubovi_img, cmap="gray")
plt.show()   

