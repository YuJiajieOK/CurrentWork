import sys
import os
import numpy
import pandas
import scipy.ndimage
import matplotlib
from pydicom import dicomio
import pylab
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def resample(one_slice, one_patient, sliceThickness, new_spacing = [1,1,1]):
#    new_spacing = [1,1,1]
    spacing = numpy.array([sliceThickness] + one_patient[0].PixelSpacing, dtype=numpy.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = one_slice.ImagePositionPatient * resize_factor
    new_shape = numpy.round(new_real_shape)
    real_resize_factor = new_shape / one_slice.ImagePositionPatient
    new_spacing = spacing / real_resize_factor
    one_slice = scipy.ndimage.interpolation.zoom(one_patient, real_resize_factor, mode='nearest')
    return one_slice

def reshape(m):
    n_row = len(m)
    n_col = len(m[0])
    # arr = [0]*1[0]*(n_row*n_col)
    arr = [0.0]*n_row*n_col
    index = 0
    for i in range(n_row):
        for j in range(n_col):
            arr[index]= float(m[i][j])
            index = index+1
    return arr

# n_nodes_hl1 = 200
# n_nodes_hl2 = 200
# n_nodes_hl3 = 180

n_classes = 2

batch_size = 100

# x = tf.placeholder('float', [None,512*512])
x = tf.placeholder('float')
y = tf.placeholder('float')


def jiajie_NN_model(CTdata,n_nodes_hl):
    # inputs*weight+biases
    hidden_1_layer = {'weight': tf.Variable(tf.random_normal([512*512,n_nodes_hl])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl]))}
    hidden_2_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl,n_nodes_hl])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl]))}
    hidden_3_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl,n_nodes_hl])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl]))}
    output_layer = {'weight': tf.Variable(tf.random_normal([n_nodes_hl, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(CTdata, hidden_1_layer['weight']), hidden_1_layer['biases'])
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['biases'])
    l2 = tf.nn.sigmoid(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['biases'])
    l3 = tf.nn.sigmoid(l3)

    outputs = tf.matmul(l3, output_layer['weight'])+output_layer['biases']
    outputs = tf.nn.sigmoid(outputs)

    return outputs



datadir = 'data'
patients = os.listdir(datadir)

for patient in patients:
    slicesList = os.listdir(datadir+'/'+patient)
    slices = [dicomio.read_file(datadir + '/' + '/'+ patient + '/' + s) for s in os.listdir(datadir+'/'+patient)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2])) # Sort slices based on ImagePositionPatient[2]
    slice_thickness = numpy.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    F_matrix = [[0.0]*512*512]*len(slices)
    s = 0
    for slice in slicesList:
         data = dicomio.read_file(datadir+'/'+ patient + '/'+ slice)
         T_matrix = reshape(data.pixel_array)
         F_matrix[s] = T_matrix
         s = s+1
         print(s)
    prediction = jiajie_NN_model(x,len(slices))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdadeltaOptimizer().minimize(cost)
    times_epochs = 10
    # model = tf.global_variables_initializer()
    with tf.Session() as sess:
         model = tf.global_variables_initializer()
         # sess.run(tf.initialize_all_variables())
         sess.run(model)
         # epoch_loss = [[0]*1]*len(slices)
         epoch_loss = 0
         # print(prediction)
         for epoch in range(times_epochs):
              epoch_x = F_matrix
              # epoch_y = [[0]*1]*len(slices)
              epoch_y = 0
              c = sess.run([optimizer,cost],feed_dict = {x:epoch_x,y:epoch_y})
              epoch_loss  = epoch_loss + c[1]
              # s_epochLoss = 0
              # avgEpochLoss = 0
              # for i in range(len(epoch_loss)):
              #     print(epoch_loss[i][1])
              #     s_epochLoss = epoch_loss[i][1]+s_epochLoss
              #
              # avgEpochLoss = s_epochLoss/len(epoch_loss)
              print('Epoch', epoch, 'out of', times_epochs,'loss', epoch_loss)
             # correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
             # accuracy = tf.reduce_mean(tf.cast(correct,'float'))



       # pylab.imshow(pixels,cmap=pylab.cm.bone)
       # pylab.imshow(pixels)
       # pylab.show()


#         resampled = resample(data, slices,slice_thickness, [1,1,1])

print("Done.")
