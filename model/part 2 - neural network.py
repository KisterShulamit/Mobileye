#----------------------------
# Selection of traffic lights
#---------------------------



#----------------------
# get suspects area.py
# ---------------------

from numpy import uint8
import numpy as np
# from tensorflow.keras.models import load_model
from keras.models import load_model
from PIL import Image
from numpy import array
score=0.45

# cut the image around the point to be in 81*81*3 size
def cut_img(image,point):
    left =  point[0] - 40
    top =  point[1] - 40
    right = point[0] + 41
    bottom = point[1] + 41

    if(left<=0):
        right -= left
        left=0
    if (top <= 0):
        bottom -= top
        top = 0
    if (right >=2048):
        left -= right-2048
        right = 2048
    if (bottom >= 1024):
        top -= bottom - 1024
        bottom = 1024

    image_from_array = Image.fromarray(image)
    return uint8(np.asarray(image_from_array.crop((left, top, right, bottom))))


# cut multiple images
def cut_images(image, points ):
    return [cut_img(image,point) for point in points]

#The main function
def get_traffic_light_points(image,red_x, red_y, green_x, green_y):
    print("get_traffic_light_points")
    # loading the neuron network
    loaded_model = load_model('/content/drive/MyDrive/Colab_Notebooks/model.h5')

    # get the cut images of the red points
    red_points=list(zip(red_x, red_y))
    images_from_red_points=array(cut_images(image,red_points))

    # mark as -1 the images that the score from the neuron network is low
    red_predictions = loaded_model.predict(images_from_red_points)[:,1]
    for i in range (len(red_points)):
        if red_predictions[i]<score:
            red_x[i]=-1
            red_y[i]=-1

    # get the cut images of the green points
    green_points = list(zip(green_x, green_y))
    images_from_green_points = array(cut_images(image, green_points))

    # mark as -1 the images that the score from the neuron network is low
    green_predictions = loaded_model.predict(images_from_green_points)[:, 1]
    for i in range(len(green_points)):
        if green_predictions[i] < score:
            green_x[i] = -1
            green_y[i] = -1

    # remove the not-traffic-lights points
    red_x=red_x[np.where(red_x != -1)]
    red_y=red_y[np.where(red_y != -1)]
    green_x=green_x[np.where(green_x != -1)]
    green_y=green_y[np.where(green_y != -1)]

    return red_x, red_y, green_x, green_y



