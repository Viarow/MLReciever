import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.simulated_dataset import QAM_Dataset, QAM_Dataset_Constant
from dataset.mapping import QAM_Mapping
from utils import *
import numpy as np
from network.classics import zero_forcing, MMSE
import argparse
import logging
import os
import time
import matplotlib.pyplot as plt

""" Test zero-forcing and minimum mean-sqaured error """

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
    parser.add_argument('--test_size', type=int, default=20, help="Size of testing dataset")
    parser.add_argument('--log_dir', type=str, help='The folder to store logging file, model state dict and figures')
    args = parser.parse_args()
    return args


def test_classic(args):

    start_time = time.time()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(args.log_dir,'output.log'))
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    params = {
        'NR': args.BaseStation * args.Antenna,
        'NT': args.User,
        'modulation': args.modulation,
        'channel': args.channel
    }

    SNRdB_range = np.linspace(args.SNRdB_min, args.SNRdB_max, args.test_size)
    testset = QAM_Dataset_Constant(params, SNRdB_range)
    #testset = QAM_Dataset(params, SNRdB_range)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    SER_zf = []
    SER_mmse = []
    BER_zf = []
    BER_mmse = []
    mod_n = int(args.modulation.split('_')[1])
    mapping = QAM_Mapping(args.modulation)
    
    with torch.no_grad():
        for i, data_blob in enumerate(testloader, 0):
            indices = data_blob['indices']
            SNRdB = data_blob['SNRdB']
            x = data_blob['x']
            y = data_blob['y']
            H = data_blob['H']
            noise_sigma = data_blob['noise_sigma']
            constellation = get_QAMconstellation(mod_n)

            xhat_zf = zero_forcing(y, H)
            xhat_mmse = MMSE(y, H, noise_sigma)
            indices_zf = QAM_demodulate(xhat_zf, constellation)
            indices_mmse = QAM_demodulate(xhat_mmse, constellation)
            ser_zf = 1. - batch_symbol_acc(indices, indices_zf)
            SER_zf.append(ser_zf)
            ser_mmse = 1. - batch_symbol_acc(indices, indices_mmse)
            SER_mmse.append(ser_mmse)
            print("SNR: {:.4f}dB  SER-ZF: {:.4f} SER-MMSE: {:.4f}".format(SNRdB.item(), ser_zf, ser_mmse))
            logger.info("SNR: {:.4f}dB  SER-ZF: {:.4f} SER-MMSE: {:.4f}".format(SNRdB.item(), ser_zf, ser_mmse))

            x_bitseq = mapping.idx_to_bits(indices[0])
            zf_bitseq = mapping.idx_to_bits(indices_zf[0])
            mmse_bitseq = mapping.idx_to_bits(indices_mmse[0])
            ber_zf = 1. - bit_accuracy(x_bitseq, zf_bitseq)
            ber_mmse = 1. - bit_accuracy(x_bitseq, mmse_bitseq)
            BER_zf.append(ber_zf)
            BER_mmse.append(ber_mmse)
            print("SNR: {:.4f}dB  BER-ZF: {:.4f} BER-MMSE: {:.4f}".format(SNRdB.item(), ber_zf, ber_mmse))
            logger.info("SNR: {:.4f}dB  BER-ZF: {:.4f} BER-MMSE: {:.4f}".format(SNRdB.item(), ber_zf, ber_mmse))

    SER_zf = np.asarray(SER_zf)
    SER_mmse = np.asarray(SER_mmse)
    BER_zf = np.asarray(BER_zf)
    BER_mmse = np.asarray(BER_mmse)

    fig_s, ax_s = plt.subplots()
    ax_s.plot(SNRdB_range, SER_zf, label='zero-forcing')
    ax_s.plot(SNRdB_range, SER_mmse, label='MMSE')
    ax_s.set_yscale('log')
    #ax_s.set_ylim([-0.1, 1])
    leg_s = ax_s.legend()
    ax_s.legend(loc='upper right', frameon=True)
    ax_s.set_xlabel('SNR(dB)')
    ax_s.set_ylabel('SER')
    title_s = '{:d}I{:d}O '.format(params['NT'], params['NR']) + params['modulation']
    ax_s.set_title(title_s)
    plt.grid()
    plt.tight_layout()
    path_s = os.path.join(args.log_dir, 'SER_constant.png')
    fig_s.savefig(path_s, dpi=fig_s.dpi)

    fig_b, ax_b = plt.subplots()
    ax_b.plot(SNRdB_range, BER_zf, label='zero-forcing')
    ax_b.plot(SNRdB_range, BER_mmse, label='MMSE')
    ax_b.set_yscale('log')
    #ax_b.set_ylim([-0.1, 1])
    leg_b = ax_b.legend()
    ax_b.legend(loc='upper right', frameon=True)
    ax_b.set_xlabel('SNR(dB)')
    ax_b.set_ylabel('BER')
    title_b = '{:d}I{:d}O '.format(params['NT'], params['NR']) + params['modulation']
    ax_b.set_title(title_b)
    plt.grid()
    plt.tight_layout()
    path_b = os.path.join(args.log_dir, 'BER_constant.png')
    fig_b.savefig(path_b, dpi=fig_b.dpi)

    print(path_s + " and " + path_b + " saved.")
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Testing finished in {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    args = parse_args()
    test_classic(args)
