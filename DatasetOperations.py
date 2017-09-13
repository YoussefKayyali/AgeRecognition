import os
from PIL import Image
from pathlib import Path
import tensorflow as tf
import numpy as np
X = []

pathphotos = Path("/home/ahmed/AgeRecognition/TrainData").glob('**/*.jpg')

for ppath in pathphotos:
    #convert to numpy array
    i=np.asarray(Image.open(str(ppath)))
    #convert to tensor
    x=tf.convert_to_tensor(i)
    #X is a list of tensors
    X.append(x)

