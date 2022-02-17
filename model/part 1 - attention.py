# --------------------------------------------------------------
# Logical_unit
# --------------------------------------------------------------


#------------------------
# Identify light sources
#------------------------


#----------
# kerne.py
#----------

from PIL import Image

def build_kernel():

    print("build_kernel")

    # """The kernel that is built is an image of 13 pixels on 13 pixeles.
    # The image  is a blurred white circle in the center surrounded by a black square that creates a strong contrast,
    #  that will be suited for detecting a traffic light.
    #  It will be looking for a circle shaped object with a strong contrast frame
    #  :return Kernel: 2D array with sum 0
    #  """

    circle_img = Image.open('/content/drive/MyDrive/Colab_Notebooks/light.png').convert('L')
    kernel = np.asarray(circle_img)
    kernel = kernel.astype(np.float32)
    kernel -= 100
    sum_circle = np.sum(kernel)
    area = circle_img.width * circle_img.height
    kernel -= (sum_circle / area)
    max_kernel = np.max(kernel)
    kernel /= max_kernel

    return kernel



class Kernel:

    __instance=None
    @staticmethod

    def getInstance():
        if not Kernel.__instance:
            Kernel()
        return Kernel.__instance

    def __init__(self):
        if Kernel.__instance:
            raise Exception("Kernel is singleton class,"
                            " Instead of initial new Instance you"
                            " can use the getInstance() method.")
        Kernel.__instance = self
        self.__kernel=build_kernel()

    def get_kernel(self):
        return self.__kernel



#--------------
# attention.py
#--------------
try:
    import test
    import os
    import json
    import glob
    import argparse
    import cv2

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    from scipy import misc
    from PIL import Image
    import matplotlib.pyplot as plt
    from scipy.spatial import distance
    from scipy.spatial.distance import pdist, squareform
    # from Logical_unit.Identify_light_sources.kernel import Kernel
except ImportError:
    print("Need to fix the installation")
    raise


def find_tfl_lights(c_image: np.ndarray, kernel):
    """    this function receives the image that we will be searching on and the current used kernel
    will return the coordinates in the image of all traffic lights
    :return - tuple of X and Y values
    """

    # get's the red layer of the picture and the green layer of the picture
    red_matrix, green_matrix = np.array(c_image)[:, :, 0], np.array(c_image)[:, :, 1]

    new_red = sg.convolve(red_matrix, kernel, mode='same')
    new_green = sg.convolve(green_matrix, kernel, mode='same')

    # filters to get the max match in each area of the green and red after doing the convolvation
    red_max = maximum_filter(new_red, size=250)
    green_max = maximum_filter(new_green, size=250)

    red_max_point = red_max == new_red
    green_max_point = green_max == new_green

    y_red, x_red = np.where(red_max_point)
    y_green, x_green = np.where(green_max_point)

    # In assumption that there are no traffic light lower than 40 or higher than 1000 we will remove
    #  the red coordinates that are out of this range
    for index in range(len(x_red)):
        if y_red[index] < 40 or y_red[index] > 1000:
            # marking the unnecessary indexes as -1 in order to delete them
            x_red[index] = -1
            y_red[index] = -1
    y_red = np.delete(y_red, np.where(y_red == -1))
    x_red = np.delete(x_red, np.where(x_red == -1))
    # In assumption that there are no traffic light lower than 40 or higher than 1000 we will remove
    #  the green coordinates that are out of this range
    for index in range(len(x_green)):
        if y_green[index] < 40 or y_green[index] > 1000:
            # marking the unnecessary indexes as -1 in order to delete them
            x_green[index] = -1
            y_green[index] = -1

    y_green = np.delete(y_green, np.where(y_green == -1))
    x_green = np.delete(x_green, np.where(x_green == -1))

    return x_red, y_red, x_green, y_green

def find_tfl_lights_in_image(image):
    print("find_tfl_lights_in_image")

    my_kernel=Kernel.getInstance()
    kernel = my_kernel.get_kernel()

    # opening the image as cv
    small_img = cv2.pyrDown(image.copy())

    # finding the coordinates of the green and red traffic lights of the original image
    red_x_small, red_y_small, green_x_small, green_y_small = find_tfl_lights(image , kernel)


    # finding the coordinates of the green and red traffic lights of the reduced image
    red_x_big, red_y_big, green_x_big, green_y_big = find_tfl_lights(small_img, kernel)


    # converting the returned coordinates to numpy type
    #  resizing the fixels of the detected cordinations of the reduced image
    # that the detected cordinates will be right on the original picture
    rx_small = np.array(red_x_small)
    ry_small = np.array(red_y_small)
    rx_big = np.array(red_x_big * 2)
    ry_big = np.array(red_y_big * 2)
    gx_small = np.array(green_x_small)
    gy_small = np.array(green_y_small)
    gx_big = np.array(green_x_big * 2)
    gy_big = np.array(green_y_big * 2)


    # concatenating the same type coordinates of the reduced image and the original one.
    red_x = np.concatenate([rx_small, rx_big])
    red_y = np.concatenate([ry_small, ry_big])
    green_x = np.concatenate([gx_small, gx_big])
    green_y = np.concatenate([gy_small, gy_big])


    return red_x,red_y,green_x,green_y







