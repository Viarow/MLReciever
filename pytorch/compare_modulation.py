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
    parser.add_argument('--modulation', type=str, default='QAM_16', help='Modulation scheme')
    parser.add_argument('--channel', type=str, default='AWGN', help='Channel Type')
    parser.add_argument('--amplifier', type=str, default='WienerHammerstein', help='Amplifier Type')
    parser.add_argument('--satlevel', type=float, default=0.5, help='Saturation level of high power amplifier')
    parser.add_argument('--SNRdB_min', type=float, default=5, help='Minimum SNR expressed in dB')
    parser.add_argument('--SNRdB_max', type=float, default=5, help='Maximum SNR expressed in dB')
    parser.add_argument('--test_size', type=int, default=20, help="Size of testing dataset")
    parser.add_argument('--batch_size_test', type=int, default=100, help="Test batch size to compute error.")
    parser.add_argument('--cuda', type=bool, default=True, help='Set true when cuda is available')
    args = parser.parse_args()
    return args


MMNet_INFO = {
    'linear_name':'MMNet_linear',
    'denoiser_name': 'MMNet_Denoiser',
    'num_layers': 10,
    'QAM_16':{
        'ckpt_linear': 'experiments_AWGN/SISO_QAM16_AWGN_LINEAR_MMNet_500epochs/MMNet_10layers_epoch500.pth',
        'ckpt_nonlinear': 'experiments_AWGN/SISO_QAM16_AWGN_NONLINEAR_MMNet_500epochs/MMNet_10layers_epoch500.pth'
    },
    'QAM_64':{
        'ckpt_linear': 'experiments_AWGN/SISO_QAM64_AWGN_LINEAR_MMNet_500epochs/MMNet_10layers_epoch500.pth',
        'ckpt_nonlinear': 'experiments_AWGN/SISO_QAM64_AWGN_NONLINEAR_MMNet_500epochs/MMNet_10layers_epoch500.pth'
    },
    'QAM_256':{
        'ckpt_linear': 'experiments_AWGN/SISO_QAM256_AWGN_LINEAR_MMNet_500epochs/MMNet_10layers_epoch500.pth',
        'ckpt_nonlinear': 'experiments_AWGN/SISO_QAM256_AWGN_NONLINEAR_MMNet_500epochs/MMNet_10layers_epoch500.pth'
    }
}


FCNet_INFO = {
    'dropout': False,
    'upstream': 1,
    'downstream': 1,
    'p': 0.,
    'QAM_16': {
        'ckpt_linear': 'experiments_AWGN/SISO_QAM16_AWGN_LINEAR_FCNet_500epochs/upstream1_downstream1_epoch500.pth',
        'ckpt_nonlinear': 'experiments_AWGN/SISO_QAM16_AWGN_NONLINEAR_FCNet_500epochs/upstream1_downstream1_epoch500.pth'
    },
    'QAM_64': {
        'ckpt_linear': 'experiments_AWGN/SISO_QAM64_AWGN_LINEAR_FCNet_500epochs/upstream1_downstream1_epoch500.pth',
        'ckpt_nonlinear': 'experiments_AWGN/SISO_QAM64_AWGN_NONLINEAR_FCNet_500epochs/upstream1_downstream1_epoch500.pth'
    },
    'QAM_256': {
        'ckpt_linear': 'experiments_AWGN/SISO_QAM256_AWGN_LINEAR_FCNet_500epochs/upstream1_downstream1_epoch500.pth',
        'ckpt_nonlinear': 'experiments_AWGN/SISO_QAM256_AWGN_NONLINEAR_FCNet_500epochs/upstream1_downstream1_epoch500.pth'
    }
}

AMP_INFO = {
    'order': 2,
    'coefficients': [1.0, -0.01]
}


def test_MMNet(args, network_type, model, testloader):
    SNR_list = []
    SER_list = []
    BER_list = []
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

            xhat_list = model(x, y, H, noise_sigma)
            xhat = xhat_list[-1]
            indices_hat = QAM_demodulate(xhat, constellation)
            SNR_list.append(SNRdB[0])
            SER = 1. - batch_symbol_acc(indices, indices_hat)
            SER_list.append(SER)
            
            BER = 1. - batch_bit_acc(args, indices, indices_hat)
            BER_list.append(BER)
            info_format = network_type + ": SNRdB: {:.2f}, SER: {:.3f}, BER: {:.3f}"
            print(info_format.format(SNRdB[0], SER, BER))

    return np.asarray(SNR_list), np.asarray(SER_list), np.asarray(BER_list)


def test_FCNet(args, network_type, model, testloader):
    SNR_list = []
    SER_list = []
    BER_list = []
    # TODO: consider PSK here later
    mod_n = int(args.modulation.split('_')[1])
    mapping = QAM_Mapping(args.modulation)
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
            
            xhat_list = model(y)
            xhat = xhat_list[-1]
            indices_hat = QAM_demodulate(xhat, constellation)
            SNR_list.append(SNRdB[0])
            SER = 1. - batch_symbol_acc(indices, indices_hat)
            SER_list.append(SER)
            
            BER = 1. - batch_bit_acc(args, indices, indices_hat)
            BER_list.append(BER)
            info_format = network_type + ": SNRdB: {:.2f}, SER: {:.3f}, BER: {:.3f}"
            print(info_format.format(SNRdB[0], SER, BER))
    
    return np.asarray(SNR_list), np.asarray(SER_list), np.asarray(BER_list)



def test_one_modulation(model, testloader_linear, testloader_nonlinear, info):

    SER_results = {}
    BER_results = {}

    # model trained on linear data --> test on linear data
    ReceiverModel.load_state_dict(torch.load(info['linear']))
    print(info['linear'] + " loaded.")
    SNR_array, SER_array, BER_array = test_MMNet(args, 'MMNet', ReceiverModel, testloader_linear)
    SER_results.update({'LMLD': SER_array})
    BER_results.update({'LMLD': BER_array})

    # model trained on linear data --> test on non-linear data
    SNR_array, SER_array, BER_array = test_MMNet(args, 'MMNet', ReceiverModel, testloader_nonlinear)
    SER_results.update({'LMND': SER_array})
    BER_results.update({'LMND': BER_array})

    # model trained on non-linear data --> test on non-linear data
    ReceiverModel.load_state_dict(torch.load(info['nonlinear']))
    print('linear'] + " loaded.")
    SNR_array, SER_array, BER_array = test_MMNet(args, 'MMNet', ReceiverModel, testloader_nonlinear)
    SER_results.update({'NMND': SER_array})
    BER_results.update({'NMND': BER_array})

    return SNR_array, SER_results, BER_results


def plot_subplots(SNRdB_range, record, ylabel, path):
    sns.set_style('whitegrid')
    modulation = ['QAM_16', 'QAM_64', 'QAM_64']
    counter = 0
    fig, axs = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    for ax in axs.flat:
        ax.set_title(modulation[counter])
        results = record[modulation[counter]]
        ax.plot(SNRdB_range, results['LMLD'], linewidth=2, label='train with linearity')
        ax.plot(SNRdB_range, results['NMLD'], linewidth=2, label='test with non-linearity')
        ax.plot(SNRdB_range, results['NMND'], linewidth=2, label='re-train with non-linearity')
        ax.set_yscale('log')
        ax.set_ylim(1, 1e-4)
        ax.set_xlabel('SNR(dB)')
        ax.set_ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3)
    fig.savefig(path, dpi=fig.dpi)
    print(path + ' saved.')


def compare_MMNet_QAM(args, fig_dir):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
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
        'amplifier': args.amplifier,
        'batch_size': args.batch_size_test,
        'constellation': constellation,
        'cuda': args.cuda
    }

    ReceiverModel = MMNet(params, MMNet_INFO['linear_name'], MMNet_INFO['denoiser_name'],
                    MMNet_INFO['num_layers'])
    if args.cuda:
        ReceiverModel = ReceiverModel.cuda()

    SNRdB_range_test = np.linspace(args.SNRdB_min, args.SNRdB_max, args.test_size)
    SNRdB_range_test = np.repeat(SNRdB_range_test, args.batch_size_test, axis=0)
    testset_linear = QAM_Dataset(params, SNRdB_range_test)
    testloader_linear = DataLoader(testset_linear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)
    params.update({'amplifier':args.amplifier, 'order':2, 'coefficients': [1.0, -0.01]})
    testset_nonlinear = QAM_Dataset_Nonlinear(params, SNRdB_range_test)
    testloader_nonlinear = DataLoader(testset_nonlinear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)

    SER_record = {}
    BER_record = {}
    
    # QAM_16
    SNR_array, SER_results, BER_results = test_one_modulation(ReceiverModel, testloader_linear, testloader_nonlinear, MMNet_INFO['QAM_16'])
    SER_record.update({'QAM_16': SER_results})
    BER_record.update({'QAM_16': BER_results})

    # QAM_64
    SNR_array, SER_results, BER_results = test_one_modulation(ReceiverModel, testloader_linear, testloader_nonlinear, MMNet_INFO['QAM_64'])
    SER_record.update({'QAM_64': SER_results})
    BER_record.update({'QAM_64': BER_results})

    # QAM_256
    SNR_array, SER_results, BER_results = test_one_modulation(ReceiverModel, testloader_linear, testloader_nonlinear, MMNet_INFO['QAM_256'])
    SER_record.update({'QAM_256': SER_results})
    BER_record.update({'QAM_256': BER_results})

    path = os.path.join(fig_dir, 'MMNet_QAM16_64_256_SER.png')
    plot_subplots(SNR_array, SER_record, 'SER', path)
    
    path = os.path.join(fig_dir, 'MMNet_QAM16_64_256_BER.png')
    plot_subplots(SNR_array, BER_record, 'BER', path)