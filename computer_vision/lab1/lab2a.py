import matplotlib.pyplot as plt
import scipy.misc as misc
import numpy as np
import math


def recover_affine_diamond(Hs, Ws, Hd, Wd):
    
    A_m = np.array([
        [Ws/2, 0, 0, 0, 1, 0],
        [0, 0, Ws/2, 0, 0, 1],
        [Ws/2, Hs/2, 0, 0, 1, 0],
        [0, 0, Ws/2, Hs/2, 0, 1],
        [Ws/2, Hs, 0, 0, 1, 0],
        [0, 0, Ws/2, Hs, 0, 1]
    ])
    b = np.array([[0], [0], [0], [Hd/2], [0], [Hd]])

    x = np.linalg.solve(A_m, b)

    affine_matrix = np.array([[x[0], x[1]], [x[2], x[3]]])
    b_matrix = np.array([x[4], x[5]])

    return affine_matrix, b_matrix

def affine_nn(Is, A, b, Hd, Wd):

    if len(Is.shape) == 3:
            color_channels = Is.shape[2]
    else:
        color_channels = 1 

    Id = np.zeros((Hd, Wd, color_channels), dtype=Is.dtype)
    
    for current_row in range(Hd):
        for current_column in range(Wd):
            source_coords = np.matmul(A, np.array([[current_column], [current_row]])) + b
            source_x = math.floor(source_coords[0])
            source_y = math.floor(source_coords[1])
            Id[current_row, current_column] = Is[source_y, source_x]
    return Id

def affine_bilin(Is, A, b, Hd, Wd):

    if len(Is.shape) == 3:
        color_channels = Is.shape[2]
    else:
        color_channels = 1 

    Id = np.zeros((Hd, Wd, color_channels), dtype=Is.dtype)
    
    for current_row in range(Hd):
        for current_column in range(Wd):
            source_coords = np.matmul(A, [current_column, current_row]) + b
            x_new, y_new = source_coords[0], source_coords[1]
            x_rounded, y_rounded = math.floor(x_new), math.floor(y_new)
            x_delta, y_delta = x_new - x_rounded, y_new - y_rounded
            Id[current_row, current_column] = Is[y_rounded, x_rounded] * (1 - x_delta) * (1 - y_delta)  + Is[y_rounded, x_rounded + 1] * x_delta * (1 - y_delta) + Is[y_rounded + 1, x_rounded] * (1 - x_delta) * y_delta + Is[y_rounded + 1, x_rounded + 1] * x_delta * y_delta 
    return Id

Is = misc.face()
Is = np.asarray(Is)

Hd, Wd = 200, 200
A,b = recover_affine_diamond(Is.shape[0],Is.shape[1], Hd,Wd)
print(A)

Id1 = affine_nn(Is, A, b, Hd, Wd)
Id2 = affine_bilin(Is, A, b, Hd, Wd)

sd = np.sqrt(np.mean((Id1 - Id2)**2))
print("Standardna devijacija između Id1 i Id2: {}".format(sd))

fig = plt.figure()
if len(Is.shape) == 2:
    plt.gray()
for i, im in enumerate([Is, Id1, Id2]):
    fig.add_subplot(1, 3, i + 1)
    plt.imshow(im.astype(int))
plt.show()