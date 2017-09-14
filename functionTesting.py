import sys
import os
import tensorflow as tf
import numpy as np
from pydicom import dicomio

with tf.Session() as sess:


# x = [[0]*1]*512
# x = tf.Variable(tf.zeros([1,512]))
     x =tf.Variable( tf.random_normal([1, 512]))
     init_op = tf.global_variables_initializer()
     sess.run(init_op)
     print(sess.run(x[0][0]))





# datadir = 'data'
# patients = os.listdir(datadir)
#
# for patient in patients:
#     slicesList = os.listdir(datadir+'/'+patient)
#     slices = [dicomio.read_file(datadir + '/' + '/'+ patient + '/' + s) for s in os.listdir(datadir+'/'+patient)]
#
#     x = tf.placeholder('float')
#     y = tf.matmul(x, x)
#
#     with tf.Session() as sess:
#         print(sess.run(y))  # ERROR: will fail because x was not fed.
#
#         rand_array = np.random.rand(1024, 1024)
#         print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.


