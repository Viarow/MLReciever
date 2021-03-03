import tensorflow as tf
import numpy as np
from utils import *
import scipy.linalg as linalg
import os

''' How to use:
    generator = Generator(params, batch_size)
    constellation = generator.QAM_N_const()
    indices = generator.QAM_N_ind()
    xBatch = indices
    sBatch = generator.modulate(indices)
    yBatch, HBatch, complexnoise_sigma, actual_snrdB = generator.channel(sBatch, snr_db_min, snr_db_max, H=None, 
                                                              dataset_flag=False)
    # xBatch.shape = [batch_size, 2*NT]
    # sBatch.shape = [batch_size, 2*NT]
    # HBatch.shape = [batch_size, 2*NR, 2*NT]
    # yBatch.shape = [batch_size, 2*NR]

    From the neural network we will get:
    # sHatBatch.shape = [batch_size, 2*NT]
    xHatBatch = demodulate(sHatBatch, constellation)
    acc = accuracy(xBatch, xHatBatch)
'''

class Generator(object):
    def __init__(self, params, batch_size):
        modulation = params['modulation']
        self.batch_size = batch_size
        if params['data']:
            self.Hdataset_powerdB = params['Hdataset_powerdB']
        else:
            self.Hdataset_powerdB = np.inf
        self.NT = params['NT']
        self.NR = params['NR']
        self.mod_scheme = modulation.split('_')[0]
        self.alpha4 = 1.
        self.alpha16 = 1.
        self.alpha64 = 1.

        self.mix_mapping = None

        if self.mod_scheme == 'QAM':
            self.mod_n = int(modulation.split('_')[1])
        elif self.mod_scheme == 'MIX':
            self.mod_n = int(modulation.split('_')[1])
        else:
            print("Modulation Type not supported.")

        self.constellation = self.QAM_N_const()

    
    def QAM_N_const(self, n=None):
        if n==None:
            n = self.mod_n
        constellation = np.linspace(-np.sqrt(n)+1, np.sqrt(n)-1, int(np.sqrt(n)))
        alpha = np.sqrt((constellation ** 2).mean())
        constellation = constellation / (alpha * np.sqrt(2))
        constellation = tf.Variable(constellation, trainable=False, dtype=tf.float32)
        return constellation

    
    def QAM_N_ind(self):
        indices_QAM = tf.random.uniform(shape=[self.batch_size, 2*self.NT],
                                        minval=0, maxval=np.sqrt(self.mod_n),
                                        dtype=tf.int32)
        if self.mod_scheme == 'MIX':
            # mod_names: 0 --> QAM4, 1 --> QAM16, 2 --> QAM64
            mod_names = tf.random.uniform(shape=[self.batch_size, self.NT],
                                          minval=0, maxval=3, dtype=tf.int32)
            mod_names = nf.tile(mod_names, [1,2])
            mapping = tf.one_hot(mod_names, depth=3, dtype=tf.int32)
            indices_QAM64 = tf.random.uniform(
                shape=[self.batch_size, 2*self.NT,1], 
                minval=0, 
                maxval=8,
                dtype=tf.int32)
            indices_QAM16 = tf.random.uniform(
                shape=[self.batch_size, 2*self.NT,1], 
                minval=0, 
                maxval=4,
                dtype=tf.int32)
            indices_QAM4 = tf.random.uniform(
                shape=[self.batch_size, 2*self.NT,1], 
                minval=0, 
                maxval=2,
                dtype=tf.int32)
            indices_QAM = mapping * tf.concat([indices_QAM64,tf.gather([0,2,5,7],indices_QAM16),indices_QAM4*7], axis=2)
            indices_QAM = tf.reduce_sum(indices_QAM, axis=2)
            self.mix_mapping = mapping

        return indices_QAM

    
    def modulate(self, indices):
        x = tf.gather(self.constellation, indices)
        return x


    def exp_corelation(self, rho):
        range_t = np.reshape(np.arange(1, self.NT+1), (-1,1))
        range_r = np.reshape(np.arange(1, self.NR+1), (-1,1))
        Rt = rho ** (np.abs(range_t - range_t.T))
        Rr = rho ** (np.abs(range_r - range_r.T))
        R1 = linalg.sqrtm(Rr)
        R2 = linalg.sqrtm(Rt)
        return R1, R2


    def channel(self, x, snr_db_min, snr_db_max, H, dataset_flag):
        if dataset_flag:
            H = H
        else: 
            # generate iid channels
            Hr = tf.random.normal(shape=[self.batch_size, self.NR, self.NT], stddev=1./np.sqrt(2.*self.NR), dtype=tf.float32)
            Hi = tf.random.normal(shape=[self.batch_size, self.NR, self.NT], stddev=1./np.sqrt(2.*self.NR), dtype=tf.float32)
            h1 = tf.concat([Hr, -1.*Hi], axis=2)
            h2 = tf.concat([Hi, Hr], axis=2)
            H = tf.concat([h1, h2], axis=1)
            self.Hdataset_powerdB = 0.
        
        #Channel Noise
        snr_db = tf.random.uniform(shape=[self.batch_size, 1], 
                                   minval=snr_db_min, maxval=snr_db_max,
                                   dtype=tf.float32)
        if self.mod_scheme == 'MIX':
            print(self.constellation)
            powQAM4 = 10. * tf.math.log((tf.square(self.constellation[0]) + tf.square(self.constellation[7]))/2.) / tf.math.log(10.)
            powQAM16 = 10. * tf.math.log((self.constellation[0] ** 2 + self.constellation[2] ** 2 + self.constellation[5] ** 2 + self.constellation[7] ** 2)/4.)/tf.math.log(10.)
            snr_adjusments = tf.cast(self.mix_mapping, tf.float32) * [[[0,-powQAM16-7.,-powQAM4-14.]]]
            snr_adjusments = tf.reduce_sum(snr_adjusments, axis=2) 
            snr_adjusments = tf.expand_dims(snr_adjusments, axis=1)
            H = H * (10. ** (snr_adjusments/10.))
            print('seessssse', H)

        wr = tf.random.normal(shape=[self.batch_size, self.NR], stddev=1./np.sqrt(2.), dtype=tf.float32, name='w')
        wi = tf.random.normal(shape=[self.batch_size, self.NR], stddev=1./np.sqrt(2.), dtype=tf.float32, name='w')
        w = tf.concat([wr, wi], axis=1)

        #SNR
        H_powerdB = 10.*tf.math.log(tf.reduce_mean(tf.reduce_sum(tf.square(H), axis=1), axis=0)) / tf.math.log(10.)
        average_H_powerdB = tf.reduce_mean(H_powerdB)
        average_x_powerdB = 10.*tf.math.log(tf.reduce_mean(tf.reduce_sum(tf.square(x), axis=1), axis=0)) / tf.math.log(10.)
        w = w * tf.pow(10., (10.*np.log10(self.NT) + self.Hdataset_powerdB - snr_db - 10.*np.log10(self.NR))/20.)
        complexnoise_sigma = tf.pow(10., (10.*np.log10(self.NT) + self.Hdataset_powerdB - snr_db - 10.*np.log10(self.NR))/20.)

        y = batch_matvec_mul(H, x) + w
        sig_powdB = 10. * tf.math.log(tf.reduce_mean(tf.reduce_sum(tf.square(batch_matvec_mul(H,x)), axis=1), axis=0)) / tf.math.log(10.)
        noise_powedB = 10. * tf.math.log(tf.reduce_mean(tf.reduce_sum(tf.square(w), axis=1), axis=0)) / tf.math.log(10.)
        actual_snrdB = sig_powdB - noise_powedB

        if dataset_flag:
            return y, complexnoise_sigma, actual_snrdB
        else:
            return y, H, complexnoise_sigma, actual_snrdB


def produce_simulation(params, seq_len, data_dir, file_object):
    ## file_object must be opened with 'a' or 'a+'
    generator = Generator(params, seq_len)
    constellation = generator.QAM_N_const()
    indices = generator.QAM_N_ind()
    xBatch = indices
    sBatch = generator.modulate(indices)
    yBatch, HBatch, noise_sigma, SNRdB = generator.channel(sBatch, params['snr_min'], params['snr_max'], H=None, dataset_flag=False)

    filename_y = "y_NR{:d}_seqlen{:d}_snr{:.2f}to{:.2f}.npy".format(params['NR'], seq_len, params['snr_min'], params['snr_max'])
    np.save(os.path.join(data_dir, filename_y), yBatch)
    print(filename_y + " saved.")
    filename_x = "x_NT{:d}_seqlen{:d}_snr{:.2f}to{:.2f}.npy".format(params['NT'], seq_len, params['snr_min'], params['snr_max'])
    np.save(os.path.join(data_dir, filename_x), xBatch)
    print(filename_x + " saved.")
    line = filename_y + "/" + filename_x + "\n"
    file_object.write(line)


    
