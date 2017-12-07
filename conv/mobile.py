import tensorflow as tf
import os

dataPath = 'Mobile-Net'

graph = tf.default_graph()
model = tf.train.Saver()

with tf.Session() as sess:
	model.restore(sess,'resnet_v1_50.cpkpt')
