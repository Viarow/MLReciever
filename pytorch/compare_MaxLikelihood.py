import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.simulated_dataset import QAM_Dataset, QAM_Dataset_Nonlinear
from dataset.mapping import QAM_Mapping
from network.detector import FullyConnectedNet, MMNet
from network.classics import zero_forcing, MMSE
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def parse_args():
    # shared arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--BaseStation', type=int, default=1, help='Number of base stations')
    parser.add_argument('--Antenna', type=int, default=1, help='Number of receiving antennas per base stattion')
    parser.add_argument('--User', type=int, default=1, help='Number of transmitting users')
    parser.add_argument('--modulation', type=str, default='QAM', help='Modulation scheme')
    parser.add_argument('--channel', type=str, default='AWGN', help='Channel Type')
    parser.add_argument('--amplifier', type=str, default='WienerHammerstein', help='Amplifier Type')
    parser.add_argument('--SNRdB_min', type=float, default=5, help='Minimum SNR expressed in dB')
    parser.add_argument('--SNRdB_max', type=float, default=5, help='Maximum SNR expressed in dB')
    parser.add_argument('--test_size', type=int, default=20, help="Size of testing dataset")
    parser.add_argument('--batch_size_test', type=int, default=100, help="Test batch size to compute error.")
    parser.add_argument('--cuda', type=bool, default=True, help='Set true when cuda is available')
    parser.add_argument('--fig_dir', type=str, help='The folder to store figures')
    args = parser.parse_args()
    return args


AMP_INFO = {
    'order': 2,
    'coefficients': [1.0, -0.1]
}

def SISO_Maximum_Likelihood(args, network_type, testloader):
    SNR_list = []
    SER_list = []
    BER_list = []
    power_list = []
    mod_n = int(args.modulation.split('_')[1])
    with torch.no_grad():
        for i, data_blob in enumerate(testloader, 0):
            indices = data_blob['indices']
            SNRdB = data_blob['SNRdB']
            x = data_blob['x']
            y = data_blob['y']
            H = data_blob['H']
            noise_sigma = data_blob['noise_sigma']
            constellation = get_QAMconstellation(mod_n)

            avg_power = batch_signal_power(y)
            power_list.append(avg_power)

            shape = y.shape
            y = torch.reshape(y, (-1, 1))
            constellation = torch.reshape(constellation, (1, -1))
            indices_hat = torch.argmin(torch.abs(y - constellation), dim=1).type(torch.IntTensor)
            indices_hat = torch.reshape(indices_hat, shape)

            SNR_list.append(SNRdB[0])
            SER = 1. - batch_symbol_acc(indices, indices_hat)
            SER_list.append(SER)
            
            BER = 1. - batch_bit_acc(args, indices, indices_hat)
            BER_list.append(BER)
            info_format = network_type + ": SNRdB: {:.2f}, SER: {:.3f}, BER: {:.3f}"
            print(info_format.format(SNRdB[0], SER, BER))

    return np.asarray(SNR_list), np.asarray(SER_list), np.asarray(BER_list), np.asarray(power_list)


def plot_error_rate(SNR_record, ER_record, ylabel, path):
    sns.set_style('whitegrid')
    modulation = ['QAM_16', 'QAM_64', 'QAM_256']
    counter = 0
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 7), constrained_layout=False)
    for ax in axs.flat:
        ax.set_title(modulation[counter])
        SNRdB_range = SNR_record[modulation[counter]]
        results = ER_record[modulation[counter]]
        ax.plot(SNRdB_range, results['linear'], linewidth=2, label='without PA')
        ax.plot(SNRdB_range, results['non-linear'], linewidth=2, label='with PA')
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 1)
        ax.set_xlabel('SNR(dB)')
        ax.set_ylabel(ylabel)
        counter += 1

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2)
    fig.savefig(path, dpi=fig.dpi)
    print(path + ' saved.')


def plot_avg_power(SNR_record, power_record, ylabel, path):
    sns.set_style('whitegrid')
    modulation = ['QAM_16', 'QAM_64', 'QAM_256']
    counter = 0
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 7), constrained_layout=False)
    for ax in axs.flat:
        ax.set_title(modulation[counter])
        SNRdB_range = SNR_record[modulation[counter]]
        results = power_record[modulation[counter]]
        ax.plot(SNRdB_range, results['linear'], linewidth=2, label='without PA')
        ax.plot(SNRdB_range, results['non-linear'], linewidth=2, label='with PA')
        #ax.set_yscale('log')
        ax.set_ylim(0.7, 1.25)
        ax.set_xlabel('SNR(dB)')
        ax.set_ylabel(ylabel)
        counter += 1

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2)
    fig.savefig(path, dpi=fig.dpi)
    print(path + ' saved.')


def compare_MaxLikelihood_QAM(args, fig_dir):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    params = {
        'NR': args.BaseStation * args.Antenna,
        'NT': args.User,
        'modulation': 'QAM',
        'channel': args.channel,
        'amplifier': args.amplifier,
        'order': AMP_INFO['order'], 
        'coefficients': AMP_INFO['coefficients'],
        'batch_size': args.batch_size_test,
        'constellation': None,
    }

    SNR_record = {}
    SER_record = {}
    BER_record = {}
    power_record = {}

    # QAM_16
    SER_QAM16 = {}
    BER_QAM16 = {}
    power_QAM16 = {}

    SNRdB_range_test = np.linspace(args.SNRdB_min, args.SNRdB_max, args.test_size)
    SNRdB_range_test = np.repeat(SNRdB_range_test, args.batch_size_test, axis=0)
    params['modulation'] = 'QAM_16'
    args.modulation = 'QAM_16'
    mod_n = int(params['modulation'].split('_')[1])
    params['constellation'] = get_QAMconstellation(mod_n)
    testset_linear = QAM_Dataset(params, (args.test_size*args.batch_size_test), SNRdB_range_test)
    testloader_linear = DataLoader(testset_linear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)
    testset_nonlinear = QAM_Dataset_Nonlinear(params, (args.test_size*args.batch_size_test),SNRdB_range_test)
    testloader_nonlinear = DataLoader(testset_nonlinear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)

    SNR_array, SER_array, BER_array, power_array = SISO_Maximum_Likelihood(args, 'Maximum Likelihood Linear', testloader_linear)
    SER_QAM16.update({'linear': SER_array})
    BER_QAM16.update({'linear': BER_array})
    power_QAM16.update({'linear': power_array})

    SNR_array, SER_array, BER_array, power_array = SISO_Maximum_Likelihood(args, 'Maximum Likelihood Non-linear', testloader_nonlinear)
    SER_QAM16.update({'non-linear': SER_array})
    BER_QAM16.update({'non-linear': BER_array})
    power_QAM16.update({'non-linear': power_array})

    SNR_record.update({'QAM_16': SNR_array})
    SER_record.update({'QAM_16': SER_QAM16})
    BER_record.update({'QAM_16': BER_QAM16})
    power_record.update({'QAM_16': power_QAM16})

    #QAM_64
    SER_QAM64 = {}
    BER_QAM64 = {}
    power_QAM64 = {}

    SNRdB_range_test = np.linspace(args.SNRdB_min, args.SNRdB_max+10, args.test_size+5)
    SNRdB_range_test = np.repeat(SNRdB_range_test, args.batch_size_test, axis=0)

    params['modulation'] = 'QAM_64'
    args.modulation = 'QAM_64'
    mod_n = int(params['modulation'].split('_')[1])
    params['constellation'] = get_QAMconstellation(mod_n)
    testset_linear = QAM_Dataset(params, (args.test_size+5)*args.batch_size_test, SNRdB_range_test)
    testloader_linear = DataLoader(testset_linear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)
    testset_nonlinear = QAM_Dataset_Nonlinear(params, (args.test_size+5)*args.batch_size_test, SNRdB_range_test)
    testloader_nonlinear = DataLoader(testset_nonlinear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)

    SNR_array, SER_array, BER_array, power_array = SISO_Maximum_Likelihood(args, 'Maximum Likelihood Linear', testloader_linear)
    SER_QAM64.update({'linear': SER_array})
    BER_QAM64.update({'linear': BER_array})
    power_QAM64.update({'linear': power_array})

    SNR_array, SER_array, BER_array, power_array = SISO_Maximum_Likelihood(args, 'Maximum Likelihood Non-linear', testloader_nonlinear)
    SER_QAM64.update({'non-linear': SER_array})
    BER_QAM64.update({'non-linear': BER_array})
    power_QAM64.update({'non-linear': power_array})

    SNR_record.update({'QAM_64': SNR_array})
    SER_record.update({'QAM_64': SER_QAM64})
    BER_record.update({'QAM_64': BER_QAM64})
    power_record.update({'QAM_64': power_QAM64})

    #QAM_256
    SER_QAM256 = {}
    BER_QAM256 = {}
    power_QAM256 = {}

    SNRdB_range_test = np.linspace(args.SNRdB_min, args.SNRdB_max+20, args.test_size+10)
    SNRdB_range_test = np.repeat(SNRdB_range_test, args.batch_size_test, axis=0)

    params['modulation'] = 'QAM_256'
    args.modulation = 'QAM_256'
    mod_n = int(params['modulation'].split('_')[1])
    params['constellation'] = get_QAMconstellation(mod_n)
    testset_linear = QAM_Dataset(params, (args.test_size+10)*args.batch_size_test, SNRdB_range_test)
    testloader_linear = DataLoader(testset_linear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)
    testset_nonlinear = QAM_Dataset_Nonlinear(params, (args.test_size+10)*args.batch_size_test, SNRdB_range_test)
    testloader_nonlinear = DataLoader(testset_nonlinear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)

    SNR_array, SER_array, BER_array, power_array = SISO_Maximum_Likelihood(args, 'Maximum Likelihood Linear', testloader_linear)
    SER_QAM256.update({'linear': SER_array})
    BER_QAM256.update({'linear': BER_array})
    power_QAM256.update({'linear': power_array})

    SNR_array, SER_array, BER_array, power_array = SISO_Maximum_Likelihood(args, 'Maximum Likelihood Non-linear', testloader_nonlinear)
    SER_QAM256.update({'non-linear': SER_array})
    BER_QAM256.update({'non-linear': BER_array})
    power_QAM256.update({'non-linear': power_array})

    SNR_record.update({'QAM_256': SNR_array})
    SER_record.update({'QAM_256': SER_QAM256})
    BER_record.update({'QAM_256': BER_QAM256})
    power_record.update({'QAM_256': power_QAM256})

    path = os.path.join(fig_dir, 'MaxLikelihood_QAM16_64_256_SER.png')
    plot_error_rate(SNR_record, SER_record, 'SER', path)

    path = os.path.join(fig_dir, 'MaxLikelihood_QAM16_64_256_BER.png')
    plot_error_rate(SNR_record, BER_record, 'BER', path)   

    path = os.path.join(fig_dir, 'Power_QAM16_64_256.png')
    plot_avg_power(SNR_record, power_record, 'signal power', path) 


if __name__ == '__main__':
    args = parse_args()
    compare_MaxLikelihood_QAM(args, args.fig_dir)
