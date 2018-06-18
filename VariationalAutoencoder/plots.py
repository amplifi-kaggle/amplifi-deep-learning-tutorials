import numpy as np
import PIL.Image as Image

from itertools import product
def gen_tile(X, shape, img_shape=None, border=False, rescale=True):
    assert(np.prod(shape) >= len(X))
    assert((img_shape is not None) or (X.ndim is 4))
    img_shape = X.shape[1:] if img_shape is None else img_shape
    height = img_shape[0]
    width = img_shape[1]
    n_ch = 1 if len(img_shape) is 2 else img_shape[2]

    tile = np.zeros((height*shape[0], width*shape[1], n_ch), dtype='uint8')

    for (i, j) in product(range(shape[0]), range(shape[1])):
        rowind = i*shape[1] + j
        if rowind < len(X):
            cell = X[rowind].reshape((height, width, n_ch))
            cell = 255*cell if rescale else cell
            tile[i*height:(i+1)*height,j*width:(j+1)*width,0:n_ch] = cell
            if border:
                tile[(i+1)*height-1,j*width:(j+1)*width,0:n_ch] = 255
                tile[i*height:(i+1)*height,(j+1)*width-1,0:n_ch] = 255

    tile = tile[:,:,0] if n_ch is 1 else tile
    return Image.fromarray(tile)

import matplotlib.pyplot as plt
def create_fig(name, gray=True, axisoff=True):
    fig = plt.figure(name)
    if gray:
        plt.gray()
    if axisoff:
        plt.axis('off')
    return fig
