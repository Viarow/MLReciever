import tensorflow as tf
import numpy 
from utils import *
from sample_generator import Generator
from MIMO_dataset import sigs_input_fn_train, sigs_input_fn_test
from NN_detector import FCNet
import argparse
from classics import zero_forcing, MMSE


def parse_mimo_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--NT', type=int, required=True, default=16, help="Number of transmitters")
    parser.add_argument('--NR', type=int, required=True, default=64, help="Number of receivers")
    parser.add_argument('--modulation', type=str, required=True, default='QAM_16')
    parser.add_argument('--snr_min', type=float, required=True, default=4., help="Minimum SNR in dB")
    parser.add_argument('--snr_max', type=float, required=True, default=10., help="Maximum SNR in dB")
    parser.add_argument('--test_points', type=int, required=True, default=100, help="Number of testing points.")
    parser.add_argument('--learning_rate', '-lr', type=float, required=True, default=1e-3)
    parser.add_argument('--batch_size', type=int, required=True, default=2, help='training batch size')
    parser.add_argument('--seq_len', type=int, required=True, default=2, help='sequence length')
    parser.add_argument('--epochs', type=int, required=True, default=10)
    parser.add_argument('--mapping_file', type=str, required=True, default='./simulated_data/mappings.txt')
    parser.add_argument('--gpu', type=str, required=False, default='0')
    parser.add_argument('--fig_dir', type=str, required=True, default='figures/')
    #parser.add_argument('--ckpt_load', type=str, required=False, help="Path of the checkpoint to load") ## e.g. './checkpoint_dir'
    #parser.add_argument('--ckpt_save', type=str, required=False, help="Path of the checkpoint to save") ## e.g. './checkpoint_dir/MyModel.ckpt'
    parser.add_argument()
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


def train(args, model, constellation):
    loss_object = tf.keras.losses.MeanSquaredError()
    train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
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
        template = 'Epoch {:d}, Loss: {:.4f}, SER: {:.4f}'
        print(template.format(epoch, train_loss_average, SER_average))
        

def test(args, model, constellation):
    intervals = numpy.linspace(args.snr_min, args.snr_max, args.test_points+1)
    SER_list = []
    SNRdB_list = []
    for k in range(0, args.testing_points):
        params_k = {
            'modulation': args.modulation,
            'data': False,
            'NT': args.NT,
            'NR': args.NR,
            'snr_min': intervals[k],
            'snr_max': intervals[k+1]
        }
        yBatch, xBatch, HBatch, noise_sigma, SNRdB = sigs_input_fn_test(params_k, args.seq_len)

        sHatBatch_NN = model(yBatch)
        xHatBatch_NN = demodulate(sHatBatch_NN, constellation)
        SER_NN = symbol_error_rate(xBatch, xHatBatch_NN)

        sHatBatch_ZF = zero_forcing(yBatch, HBatch)
        xHatBatch_ZF = demodulate(sHatBatch_ZF, constellation)
        SER_ZF = symbol_error_rate(xBatch, xHatBatch_ZF)

        sHatBatch_MMSE = MMSE(yBatch, HBatch, noise_sigma)
        xHatBatch_MMSE = demodulate(sHatBatch_MMSE, constellation)
        SER_MMSE = symbol_error_rate(xBatch, xHatBatch_MMSE)

        SER_k = {
            'NN': SER_NN,
            'ZF': SER_ZF,
            'MMSE': SER_MMSE
        }
        SER_list.append(SER_k)
        SNRdB_list.append(SNRdB)
        template = "Neural Network: {:.4f},    Zero-Forcing: {:.4f},    Minimum Mean Squared Error: {:.4f}"
        print(template.format(SER_NN, SER_ZF, SER_MMSE))
    
    return SNRdB_list, SER_list


def main():
    args = parse_mimo_args()
    if args.gpu != None and len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    mod_n = int(args.modulation.split('_')[1])
    constellation = modulate(mod_n)
    model = FCNet(args.NR, args.NT)
    sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())
    train(args, model, constellation)
    SNRdB_list, SER_list = test(args, model, constellation)
    plot_fig(args, SNRdB_list, SER_list, args.fig_dir)


if __name__ == '__main__':
    main()
