import tensorflow as tf
import numpy as np


myint = tf.int32
myfloat = tf.float32


class CNN:

    __slots__ = ['x', 'y', 'y_onehot', 'is_training', 'logits', 'predicted_classes']

    def __init__(self, config):

        shape_x = np.array([None, 32, 32, 3])
        shape_y = np.array([None])  # not one-hot vector

        self.x = tf.placeholder(dtype=myfloat, shape=shape_x, name='x')
        self.y = tf.placeholder(dtype=myint, shape=shape_y, name='y')
        self.y_onehot = tf.one_hot(indices=tf.cast(self.y, myint), depth=10)
        self.is_training = tf.placeholder(dtype=bool, name='is_training')
        l_1 = tf.layers.conv2d(inputs=self.x,
                                    filters=32,
                                    kernel_size=[5, 5],
                                    padding='same',
                                    activation=tf.nn.relu)
        if config['is_use_bn_conv']:
            l_1 = tf.layers.batch_normalization(l_1, training=self.is_training)
        l_2 = tf.layers.max_pooling2d(inputs=l_1, pool_size=[2, 2], strides=2)
        l_3 = tf.layers.conv2d(inputs=l_2,
                                     filters=64,
                                     kernel_size=[5, 5],
                                     padding='same',
                                     activation=tf.nn.relu)
        if config['is_use_bn_conv']:
            l_3 = tf.layers.batch_normalization(l_3, training=self.is_training)
        l_4 = tf.layers.max_pooling2d(inputs=l_3, pool_size=[2, 2], strides=2)
        l_4_flat = tf.reshape(l_4, [-1, 4096])
        l_5 = tf.layers.dense(inputs=l_4_flat, units=1024, activation=tf.nn.relu)
        if config['is_use_bn_dense']:
            l_5 = tf.layers.batch_normalization(l_5, training=self.is_training)
        if config['is_use_dropout']:
            l_5 = tf.layers.dropout(inputs=l_5,
                                    rate=config['dropout_rate'],
                                    training=self.is_training)
        l_6 = tf.layers.dense(inputs=l_5, units=10)
        self.logits = l_6

        self.predicted_classes = tf.argmax(input=self.logits, axis=1, output_type=myint)
