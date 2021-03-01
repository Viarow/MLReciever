import tensorflow as tf
import numpy as np
from sample_generator import Generator

'''For training dataset, each y file (sample) contains a numpy array with shape=[seq_len, 2*NR].
   each x file (label) contains a numpy array with shape=[seq_len, 2*NT].

   For testing dataset, generate a batch of data with the specific SNR range directly,
   returns:
        H(shape=[batch_size, 2*NR, 2*NT]), 
        x(shape=[batch_size, 2*NT]),
        y(shape=[batch_size, 2*NR]),
        noise_sigma(shape=[batch_size],
        SNRdB(shape=[batch_size])'''


## training dataset
def sigs_input_fn_train(filenames, labels, perform_shuffle=False, repeat_count=1, batch_size=2):

    def _parse_function(filename, label):
        y_np = np.load(filename)
        y = tf.convert_to_tensor(y_np, np.float32)
        x_np = np.load(label)
        x = tf.convert_to_tensor(x_np, np.int32)
        return y, x

    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(batch_size)  # Batch size to use
    
    return dataset


## testing dataset, params must contain modulation, NR, NT, snr_min, snr_max
def sigs_input_fn_test(params, batch_size):
    generator = Generator(params, batch_size)
    constellation = generator.QAM_N_const()
    indices = generator.QAM_N_ind()
    xBatch = indices
    sBatch = generator.modulate(indices)
    yBatch, HBatch, noise_sigma, SNRdB = generator.channel(sBatch, params['snr_min'], params['snr_max'], H=None, dataset_flag=False)
    return yBatch, xBatch, HBatch, noise_sigma, SNRdB