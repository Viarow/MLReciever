import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


def modulate(mod_n):
    n = mod_n
    constellation = np.linspace(-np.sqrt(n)+1, np.sqrt(n)-1, int(np.sqrt(n)))
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
        #print("tf.equal(xBatch[k], xHatBatch[k]):", tf.equal(xBatch[k], xHatBatch[k]))
        ser_k = tf.reduce_sum(tf.cast(tf.equal(xBatch[k], xHatBatch[k]), tf.int32))
        ser_k = tf.cast(ser_k/NR, tf.float32)
        SER += ser_k
    
    SER = SER/batch_size
    return SER


def plot_fig(args, SNRdB, SER_list, save_path):
    SER_NN = []
    SER_ZF = []
    SER_MMSE = []
    for SER_k in SER_list:
        SER_NN.append(SER_k['NN'])
        SER_ZF.append(SER_k['ZF'])
        SER_MMSE.append(SER_k['MMSE'])

    fig, ax = plt.subplots()
    ax.scatter(SNRdB, SER_NN, c='r', alpha=0.7, label='Neural Network')
    ax.scatter(SNRdB, SER_ZF, c='b', alpha=0.7, label='Zero Forcing')
    ax.scatter(SNRdB, SER_MMSE, c='g', alpha=0.7, label='MMSE')
    leg = ax.legend()
    ax.legend(loc='lower left', frameon=True)
    plt.xlabel('SNR(dB)')
    plt.ylabel('SER')
    template = 'NT{:d}_NR{:d}_'.format(args.NT, args.NR) + args.modulation + '_epochs_' + args.epochs
    plt.grid(template)
    plt.savefig(os.path.join(save_path, template+'.png'))