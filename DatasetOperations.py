import os
from PIL import Image
from pathlib import Path
import tensorflow as tf
X = tf.Variable([350,432])

pathphotos = Path("F:\\New folder\Data").glob('**/*.jpg')

for ppath in pathphotos:
    path_of_photo = str(ppath)
    im = Image.open(path_of_photo)
    px = im.load()
    X.append(px)
    Arr=[]
    for i in range(350):
        for j in range(432):
          Arr.append(int(px[i,j]))
    X.append(Arr)

print(X)
#print (X)