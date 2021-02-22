import tensorflow as tf
import numpy as np


def modulate(mod_n):
    n = mod_n
    constellation = np.linspace(-np.sqrt(n)+1, np.sqrt(n)-1, np.sqrt(n))
    alpha = np.sqrt((constellation ** 2).mean())
    constellation = constellation / (alpha * np.sqrt(2))
    constellation = tf.Variable(constellation, trainable=False, dtype=tf.float32)
    return constellation


def demodulate(y, constellation):
    shape = tf.shape(y)
    y = tf.reshape(y, shape=[-1,1])
    constellation = tf.reshape(constellation, shape=[1, -1])
    indices = tf.cast(tf.argmin(tf.abs(y-constellation), axis=1), tf.int32)
    indices = tf.reshape(indices, shape)
    return indices

def batch_matvec_mul(A, b, transpose_a=False):
    C = tf.matmul(A, tf.expand_dims(b, axis=2), transpose_a=transpose_a)
    return tf.squeeze(C, -1)

def accuracy(x, y):
    '''Computes the fraction of elements for which x and y are equal'''
    return tf.reduce_mean(tf.cast(tf.equal(x, y), tf.float32))

def symbol_error_rate(xBatch, xHatBatch):
    batch_size = xBatch.shape[0]
    NR = xBatch.shape[1]  # 2*NR
    SER = 0.
    for k in np.arange(0, batch_size):
        ser_k = tf.reduce_sum(tf.equal(xBatch[k], xHatBatch[k]))
        ser_k = tf.cast(ser_k/NR, tf.float32)
        SER += ser_k
    
    SER = SER/batch_size
    return SER
