import os
from PIL import Image
from pathlib import Path
import tensorflow as tf
import numpy as np
X = []
y = []

pathphotos = Path("./TrainData").glob('**/*.jpg')

#extract features and labels
for ppath in pathphotos:
    #open photo as numpy array
    path = str(ppath)
    im=np.asarray(Image.open(path))
    x = []
    #create list of every pixel in a single photo
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            x.append(im[i][j])
    #save the list of photo pixels to construct a matrix num_of_training_examples*num_of_pixels
    X.append(x)
    y.append(int(path[len(path) - 8 : len(path) - 4]) - int(path[len(path) - 19 : len(path) - 15]))