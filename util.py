# Helper methods for the vehicle tracking, a number of these were developed
# as part of other projects and are re-used here.
import cv2
import matplotlib.pyplot as plt
import numpy as np

def clahe_RGB(img, clipLimit, tileGridSize):
    """ Apply Contrast-Limited Adaptive Histogram Equalization with OpenCV

    Contrast-Limited Adaptive Histogram Equalization is applied to each
    of the three color channels of an RGB image. The result is returned
    as an RGB image.

    Args:
        img: Input image  should be in RGB colorspace.
        clipLimit: Passed to cv2.createCLAHE
        tileGridSize: Passed to cv2.createCLAHE

    Returns:
        The input image  with CLAHE applied  in RGB
    """

    r, g, b = cv2.split(img)

    img_clahe   = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_clahe_r = img_clahe.apply(r)
    img_clahe_g = img_clahe.apply(g)
    img_clahe_b = img_clahe.apply(b)

    img_ret = cv2.merge((img_clahe_r,  img_clahe_g,  img_clahe_b))

    return(img_ret)

def mosaic(images, height, width, ncol, cmap=None):
    """ Produce a mosaic from the input images.

    Images are plotted one next to the other, in ncol columns.
    They will take up the space defined by the width/height
    parameters. The number of rows is determine automatically.

    Args:
    images: Images to be plotted
    height: Plotting area height
    width: Plotting area width
    ncol: Number of columns in mosaic

    Returns:
    A pyplot figure.
    """

    figure = plt.figure(figsize=(width, height))

    for i in range(0, len(images)):
        figure_ax = figure.add_subplot(np.ceil(len(images)/ncol), ncol, i+1)

        figure_ax.imshow(images[i], figure=figure, aspect='auto', interpolation='nearest', cmap=cmap)
        figure_ax.axis('off')

    figure.subplots_adjust(wspace=0, hspace=0)

    return figure
