import os
from PIL import Image
from pathlib import Path
import tensorflow as tf
import numpy as np
X = []
y = []
learning_rate = 0.1
training_epochs = 100
batch_size = 256
display_step = 1
examples_to_show = 10

pathphotos = Path("./TrainData").glob('**/*.jpg')
n_input=151200
n_hidden_1=10
n_hidden_2=5

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

x=tf.placeholder("float",[None,n_input])
input_to_hidden_encoder_w=tf.Variable(tf.random_normal([n_input,n_hidden_1]))
input_to_hidden_encoder_b=tf.Variable(tf.random_normal([n_hidden_1]))


hidden_to_hidden_encoder_w=tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2]))
hidden_to_hidden_encoder_b=tf.Variable(tf.random_normal([n_hidden_2]))


hidden_to_hidden_decoder_w=tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1]))
hidden_to_hidden_decoder_b=tf.Variable(tf.random_normal([n_hidden_1]))


hidden_to_output_decoder_w=tf.Variable(tf.random_normal([n_hidden_1,n_input]))
hidden_to_output_decoder_b=tf.Variable(tf.random_normal([n_input]))


input_to_hidden_encode=tf.matmul(x,input_to_hidden_encoder_w)+input_to_hidden_encoder_b
input_to_hidden_encode=tf.nn.sigmoid(input_to_hidden_encode)


hidden_to_hidden_encode=tf.matmul(input_to_hidden_encode,hidden_to_hidden_encoder_w)+hidden_to_hidden_encoder_b
hidden_to_hidden_encode=tf.nn.sigmoid(hidden_to_hidden_encode)


hidden_to_hidden_decode=tf.matmul(hidden_to_hidden_encode,hidden_to_hidden_decoder_w)+hidden_to_hidden_decoder_b
hidden_to_hidden_decode=tf.nn.sigmoid(hidden_to_hidden_decode)


hidden_to_output_decode=tf.matmul(hidden_to_hidden_decode,hidden_to_output_decoder_w)+hidden_to_output_decoder_b
hidden_to_output_decode=tf.nn.sigmoid(hidden_to_output_decode)




y_pred=hidden_to_output_decode
y_true=x


cost=tf.reduce_mean(tf.pow(y_true-y_pred,2))

optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init=tf.global_variables_initializer()
