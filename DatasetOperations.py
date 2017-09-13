import os
from PIL import Image
from pathlib import Path
import tensorflow as tf
import numpy as np
X = []

pathphotos = Path("/home/ahmed/AgeRecognition/TrainData").glob('**/*.jpg')
a=0
for ppath in pathphotos:
    #open photo as numpy array
    im=np.asarray(Image.open(str(ppath)))
    x = []
    #create list of every pixel in a single photo
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            x.append(im[i][j])
    #save the list of photo pixels to construct a matrix num_of_training_examples*num_of_pixels
    X.append(x)
    #convert the list to a 2d numpy array
final_array=np.array(X)

print (final_array.shape)