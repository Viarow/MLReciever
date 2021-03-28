import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset.simulated_dataset import QAM_Dataset, QAM_Dataset_Constant
from dataset.mapping import QAM_Mapping
from network.detector import MMNet
from utils import *
from tqdm import tqdm
import numpy as np
import argparse
import logging
import os
import time

""" Network: MMNet
    Data: Single subcarrier in OFDM
    Optimizer: Adam Optimizer
    Training & Testing: default testing batch size is 1
                        save checkpoints, save SER & BER for plotting after each test
"""

def parse_args():
    parser = argparse.ArgumentParser()
    # About Dataset
    parser.add_argument('--BaseStation', type=int, default=1, help='Number of base stations')
    parser.add_argument('--Antenna', type=int, default=1, help='Number of receiving antennas per base stattion')
    parser.add_argument('--User', type=int, default=1, help='Number of transmitting users')
    parser.add_argument('--modulation', type=str, default='QAM_16', help='Modulation scheme')
    parser.add_argument('--channel', type=str, default='AWGN', help='Channel Type')
    parser.add_argument('--SNRdB_min', type=float, default=5, help='Minimum SNR expressed in dB')
    parser.add_argument('--SNRdB_max', type=float, default=5, help='Maximum SNR expressed in dB')
    parser.add_argument('--train_size', type=int, default=6400, help='Size of traing dataset')
    parser.add_argument('--batch_size_train', type=int, default=64, help='Training batch size')
    parser.add_argument('--test_size', type=int, default=20, help="Size of testing dataset")
    parser.add_argument('--batch_size_test', type=int, default=100, help="Test batch size to compute error.")
    # About Network Architecture
    parser.add_argument('--linear_name', type=str, default='MMNet_linear', help='Type of linear function in each layer')
    parser.add_argument('--denoiser_name', type=str, default='MMNet_Denoiser', help='Type of denoiser function in each layer')
    parser.add_argument('--num_layers', type=int, default=10, help='Number of layers in the network')
    # About Training
    parser.add_argument('--cuda', type=bool, default=True, help='Set true when cuda is available')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='Learning rate of the optimizer')
    parser.add_argument('--epochs', type=int, default=500, help='Total amount of epochs to train')
    parser.add_argument('--test_every', type=int, default=100, help='Epoch intervals to test during training')
    parser.add_argument('--log_every', type=int, default=10, help="Step interval to record running loss in the log file")
    parser.add_argument('--log_dir', type=str, help='The folder to store logging file, model state dict and figures')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoints of the model to load')
    args = parser.parse_args()
    return args


def test(args, epoch, model, testloader, logger):
    SNR_list = []
    SER_list = []
    BER_list = []
    # TODO: consider PSK here later
    mod_n = int(args.modulation.split('_')[1])
    with torch.no_grad():
        for i, data_blob in enumerate(testloader, 0):
            indices = data_blob['indices']
            SNRdB = data_blob['SNRdB']
            if args.cuda:
                x = data_blob['x'].cuda()
                y = data_blob['y'].cuda()
                H = data_blob['H'].cuda()
                noise_sigma = data_blob['noise_sigma'].cuda()
                constellation = get_QAMconstellation(mod_n).cuda()
            else:
                x = data_blob['x']
                y = data_blob['y']
                H = data_blob['H']
                noise_sigma = data_blob['noise_sigma']
                constellation = get_QAMconstellation(mod_n)

            xhat, _ = model(x, y, H, noise_sigma)
            indices_hat = QAM_demodulate(xhat, constellation)
            SNR_list.append(SNRdB[0])
            SER = 1. - batch_symbol_acc(indices, indices_hat)
            SER_list.append(SER)
            
            BER = 1. - batch_bit_acc(args, indices, indices_hat)
            BER_list.append(BER)
            info_format = "Epoch: {:d}, SNRdB: {:.2f}, SER: {:.3f}, BER: {:.3f}"
            print(info_format.format((epoch+1), SNRdB[0], SER, BER))
            logger.info(info_format.format((epoch+1), SNRdB[0], SER, BER))

    return np.asarray(SNR_list), np.asarray(SER_list), np.asarray(BER_list)


def train(args):

    start_time = time.time()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(args.log_dir,'output.log'))
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    mod_n = int(args.modulation.split('_')[1])
    if args.cuda:
        constellation = get_QAMconstellation(mod_n).cuda()
    else:
        constellation = get_QAMconstellation(mod_n)

    params = {
        'NR': args.BaseStation * args.Antenna,
        'NT': args.User,
        'modulation': args.modulation,
        'channel': args.channel,
        'batch_size': args.batch_size_train,
        'constellation': constellation,
        'cuda': args.cuda
    }

    SNRdB_range_train = np.linspace(args.SNRdB_min, args.SNRdB_max, args.train_size)
    trainset = QAM_Dataset(params, SNRdB_range_train)
    trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=2)
    SNRdB_range_test = np.linspace(args.SNRdB_min, args.SNRdB_max, args.test_size)
    SNRdB_range_test = np.repeat(SNRdB_range_test, args.batch_size_test, axis=0)
    testset = QAM_Dataset(params, SNRdB_range_test)
    testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=2)

    ReceiverModel = MMNet(params, linear_name = args.linear_name, 
                        denoiser_name=args.denoiser_name, num_layers=args.num_layers)
    if args.checkpoint is not None:
        ReceiverModel.load_state_dict(torch.load(args.checkpoint))
    if args.cuda:
        ReceiverModel = ReceiverModel.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ReceiverModel.parameters(), lr=args.learning_rate)

    error_list =[]
    iterations = []
    losses = []

    for epoch in range(0, args.epochs):

        running_loss = 0.0
        for i, data_blob in tqdm(enumerate(trainloader, 0)):
            #indices = data_blob['indices']
            #SNRdB = data_blob['SNRdB']
            if args.cuda:
                x = data_blob['x'].cuda()
                y = data_blob['y'].cuda()
                H = data_blob['H'].cuda()
                noise_sigma = data_blob['noise_sigma'].cuda()
            else:
                x = data_blob['x']
                y = data_blob['y']
                H = data_blob['H']
                noise_sigma = data_blob['noise_sigma']

            optimizer.zero_grad()
            xhat, _ = ReceiverModel(x, y, H, noise_sigma)
            loss = criterion(xhat, x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % args.log_every == 0:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/args.log_every))
                #logger.info('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/args.log_every))
                iterations.append(epoch * args.train_size / args.batch_size_train + i + 1)
                losses.append(running_loss/args.log_every)
                #logger.info('[%d, %5d] loss: %.3f SER: %.3f' % (epoch+1, i+1, running_loss/args.log_every, symbol_error_rate/args.log_every))
                running_loss = 0.0
                symbol_acc = 0.0

        if (epoch+1) % args.test_every == 0:
            print("Testing at epoch %d" % (epoch +1))
            logger.info("Testing at epoch %d" % (epoch +1))
            SNR_array, SER_array, BER_array = test(args, epoch, ReceiverModel, testloader, logger)
            error_list.append({'epoch': (epoch+1), 'SER': SER_array, 'BER': BER_array})
            ckpt_format = 'MMNet_{:d}layers_epoch{:d}.pth'.format(args.num_layers, (epoch+1))
            ckpt_path = os.path.join(args.log_dir, ckpt_format)
            torch.save(ReceiverModel.state_dict(), ckpt_path)
            print(ckpt_path + " saved.")
            logger.info(ckpt_path + " saved.")

    # plot and save
    loss_path = plot_loss(params, args, 'MMNet', iterations, losses)
    print(loss_path + "saved.")
    logger.info(loss_path + "saved.")
    ser_path, ber_path = plot_epochs_MMNet(params, args, SNR_array, error_list)
    print(ser_path + " saved.")
    logger.info(ser_path+ " saved.")
    print(ber_path + " saved.")
    logger.info(ber_path + " saved.")

    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training and saving finished in {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    args = parse_args()
    train(args)