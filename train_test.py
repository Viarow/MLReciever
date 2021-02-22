import tensorflow as tf
import numpy 
from utils import *
from sample_generator import Generator
from MIMO_dataset import sigs_input_fn_train, sigs_input_fn_test
from NN_detector import FCNet
import argparse


def parse_mimo_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--NT', type=int, required=True, default=16, help="Number of transmitters")
    parser.add_argument('--NR', type=int, required=True, default=64, help='Number of receivers')
    parser.add_argument('--modulation', type=str, required=True, default='QAM_16')
    parser.add_argument('--snr_min', type=float, required=True, default=11., help="Minimum SNR in dB")
    parser.add_argument('--snr_max', type=float, required=True, default=17., help="Maximum SNR in dB")
    parser.add_argument('--learning_rate', '-lr', type=float, required=True, default=1e-3)
    parser.add_argument('--batch_size', type=int, required=True, default=2, help='training batch size')
    parser.add_argument('--seq_len', type=int, required=True, default=2, help='sequence length')
    parser.add_argument('--epochs', type=int, required=True, default=10)
    parser.add_argument('--mapping_file', type=str, required=False, default='./simulated_data/mappings.txt')
    parser.add_argument('--gpu', type=str, required=False, default='0')
    args = parser.parse_args()
    return args


def parse_filenames(ref_file):
    file_object = open(ref_file, 'r')
    sample_list = file_object.readlines
    filenames = []
    labels = []
    for line in sample_list:
        sample = line.split('\n')[0]
        filename = sample_list.split('/')[0]
        label = sample_list.split('/')[1]
        filenames.append(filename)
        labels.append(label)
    file_object.close()
    return filenames, labels


def train_step(model, yBatch, xBatch, constellation, loss_object, optimizer):
    with tf.GradientTape() as tape:
        sHatBatch = model(yBatch)
        xHatBatch = demodulate(sHatBatch, constellation)
        loss = loss_object(xBatch, xHatBatch)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return xHatBatch, loss


def train(args):
    if args.gpu != None and len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model = FCNet(args.NT)
    loss_object = tf.keras.losses.MeanSquaredError()
    train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    mod_n = int(args.modulation.split('_')[1])
    constellation = modulate(mod_n)
    filenames, labels = parse_filenames(args.mapping_file)
    dataset_train = sigs_input_fn_train(filenames, labels, batch_size=args.batch_size)

    for epoch in range(0, args.epochs):
        train_loss.reset_states()
        SER = []
        for yBatch, xBatch in dataset_train:
            y_shape = yBatch.shape
            yBatch = tf.reshape(yBatch, [-1, y_shape[2]])
            x_shape = xBatch.shape
            xBatch = tf.reshape(xBatch, [-1, x_shape[2]])
            xHatBatch, loss = train_step(model, yBatch, xBatch, xBatch, 
                                        constellation, loss_object, optimizer)
            train_loss(loss)
            SER.append(symbol_error_rate(xBatch, xHatBatch))
        train_loss_average = train_loss.result()
        SER_average = numpy.sum(numpy.array(SER)) / len(SER)
        template = 'Epoch {}, Loss: {:.4f}, SER: {:.4f}'
        print(template.format(epoch, train_loss_average, SER_average))
        
    