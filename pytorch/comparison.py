import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.simulated_dataset import QAM_Dataset
from dataset.mapping import QAM_Mapping
from network.detector import FullyConnectedNet, MMNet
from network.classics import zero_forcing, MMSE
from utils import *
import numpy as np
import argparse
import os


## checkpoint information
MMNet_INFO = {
    'linear_name':'MMNet_linear',
    'denoiser_name': 'MMNet_Denoiser',
    'checkpoints':{
        '5layers' : 'experiments/SISO_QAM16_AWGN_LINEAR_MMNet_500epochs/MMNet_5layers_epoch500.pth',
        '10layers' : 'experiments/SISO_QAM16_AWGN_LINEAR_MMNet_500epochs/MMNet_10layers_epoch500.pth',
        '15layers': 'experiments/SISO_QAM16_AWGN_LINEAR_MMNet_500epochs/MMNet_15layers_epoch500.pth'
    }
} 

FCNet_INFO = {
    '1layer' : {
        'dropout': False,
        'upstream': 0,
        'downstream': 0,
        'p': 0.,
        'checkpoint': 'experiments/10I10O_QAM16_AWGN_LINEAR_FCNet_500epochs/upstream0_downstream0_epoch500.pth'
    },
    '3layers' : {
        'dropout': False,
        'upstream': 1,
        'downstream': 1,
        'p': 0.,
        'checkpoint': 'experiments/10I10O_QAM16_AWGN_LINEAR_FCNet_500epochs/upstream1_downstream1_epoch500.pth'
    },
    '5layers' : {
        'dropout': False,
        'upstream': 2,
        'downstream': 2,
        'p': 0.,
        'checkpoint': 'experiments/10I10O_QAM16_AWGN_LINEAR_FCNet_500epochs/upstream2_downstream2_epoch500.pth'
    },
    '10layers' : {
        'dropout': False,
        'upstream': 4,
        'downstream': 5,
        'p': 0.,
        'checkpoint': 'experiments/10I10O_QAM16_AWGN_LINEAR_FCNet_500epochs/upstream4_downstream5_epoch500.pth'
    },
    '15layers' : {
        'dropout': False,
        'upstream': 7,
        'downstream': 7,
        'p': 0.,
        'checkpoint': 'experiments/10I10O_QAM16_AWGN_LINEAR_FCNet_500epochs/upstream7_downstream7_epoch500.pth'
    }
}

MIXED_INFO = {
    'MMNet': {
        'linear_name':'MMNet_linear',
        'denoiser_name': 'MMNet_Denoiser',
        'num_layers': 10,
        'checkpoint': 'experiments/SISO_QAM16_AWGN_LINEAR_MMNet_500epochs/MMNet_10layers_epoch500.pth'
    },
    'FCNet': {
        'dropout': False,
        'upstream': 0,
        'downstream': 0,
        'p': 0.,
        'checkpoint': 'experiments/SISO_QAM16_AWGN_LINEAR_FCNet_500epochs/upstream0_downstream0_epoch500.pth'
    }
}

def parse_args():
    # shared arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--BaseStation', type=int, default=1, help='Number of base stations')
    parser.add_argument('--Antenna', type=int, default=1, help='Number of receiving antennas per base stattion')
    parser.add_argument('--User', type=int, default=1, help='Number of transmitting users')
    parser.add_argument('--modulation', type=str, default='QAM_16', help='Modulation scheme')
    parser.add_argument('--channel', type=str, default='AWGN', help='Channel Type')
    parser.add_argument('--SNRdB_min', type=float, default=5, help='Minimum SNR expressed in dB')
    parser.add_argument('--SNRdB_max', type=float, default=5, help='Maximum SNR expressed in dB')
    parser.add_argument('--test_size', type=int, default=20, help="Size of testing dataset")
    parser.add_argument('--batch_size_test', type=int, default=100, help="Test batch size to compute error.")
    parser.add_argument('--cuda', type=bool, default=True, help='Set true when cuda is available')
    args = parser.parse_args()
    return args


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
    
            xhat, _ = model(x, y, H, noise_sigma)
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
            
            xhat = model(y)
            indices_hat = QAM_demodulate(xhat, constellation)
            SNR_list.append(SNRdB[0])
            SER = 1. - batch_symbol_acc(indices, indices_hat)
            SER_list.append(SER)
            
            BER = 1. - batch_bit_acc(args, indices, indices_hat)
            BER_list.append(BER)
            info_format = network_type + ": SNRdB: {:.2f}, SER: {:.3f}, BER: {:.3f}"
            print(info_format.format(SNRdB[0], SER, BER))
    
    return np.asarray(SNR_list), np.asarray(SER_list), np.asarray(BER_list)


def test_classic(args, method, testloader):
    SNR_list = []
    SER_list = []
    BER_list = []
    # TODO: consider PSK here later
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

            if method == 'zero-forcing':
                xhat = zero_forcing(y, H)
            else: 
                xhat = MMSE(y, H, noise_sigma)
            indices_hat = QAM_demodulate(xhat, constellation)
            SNR_list.append(SNRdB[0])
            SER = 1. - batch_symbol_acc(indices, indices_hat)
            SER_list.append(SER)

            BER = 1. - batch_bit_acc(args, indices, indices_hat)
            BER_list.append(BER)
            info_format = method + ": SNRdB: {:.2f}, SER: {:.3f}, BER: {:.3f}"
            print(info_format.format(SNRdB[0], SER, BER))

    return np.asarray(SNR_list), np.asarray(SER_list), np.asarray(BER_list)


def compare_MMNet(args, fig_dir):
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
        'batch_size': args.batch_size_test,
        'constellation': constellation,
        'cuda': args.cuda
    }
    SNRdB_range_test = np.linspace(args.SNRdB_min, args.SNRdB_max, args.test_size)
    SNRdB_range_test = np.repeat(SNRdB_range_test, args.batch_size_test, axis=0)
    testset = QAM_Dataset(params, SNRdB_range_test)
    testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=2)

    SER_results = {}
    BER_results = {}
    checkpoints = MMNet_INFO['checkpoints']
    checkpoint_keys = list(checkpoints.keys())
    for key in checkpoint_keys:
        num_layers = int(key.split('l')[0])
        ReceiverModel = MMNet(params, linear_name=MMNet_INFO['linear_name'],
                        denoiser_name=MMNet_INFO['denoiser_name'], num_layers=num_layers)
        if args.cuda:
            ReceiverModel = ReceiverModel.cuda()
        ReceiverModel.load_state_dict(torch.load(checkpoints[key]))
        print(checkpoints[key] + ' loaded.')

        SNR_array, SER_array, BER_array = test_MMNet(args, 'MMNet', ReceiverModel, testloader)
        SER_results.update({key: SER_array})
        BER_results.update({key: BER_array})
    
    title = 'NT{:d}_NR{:d}_'.format(params['NT'], params['NR']) + params['modulation']
    SER_path = os.path.join(fig_dir, 'MMNet_SER.png')
    plot_comparison(SNR_array, SER_results, 'SER', title, SER_path)
    BER_path = os.path.join(fig_dir, 'MMNet_BER.png')
    plot_comparison(SNR_array, BER_results, 'BER', title, BER_path)
    #print(SER_path + " and " + BER_path + " saved.")


def compare_FCNet(args, fig_dir):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    mod_n = int(args.modulation.split('_')[1])

    params = {
        'NR': args.BaseStation * args.Antenna,
        'NT': args.User,
        'modulation': args.modulation,
        'channel': args.channel
    }
    SNRdB_range_test = np.linspace(args.SNRdB_min, args.SNRdB_max, args.test_size)
    SNRdB_range_test = np.repeat(SNRdB_range_test, args.batch_size_test, axis=0)
    testset = QAM_Dataset(params, SNRdB_range_test)
    testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=2)

    SER_results = {}
    SER_results_dropout = {}
    BER_results = {}
    BER_results_dropout = {}
    checkpoint_keys = list(FCNet_INFO.keys())
    for key in checkpoint_keys:
        #num_layers = int(key.split('l')[0])
        ckpt_item = FCNet_INFO[key]
        layers_dict = {
            'upstream': ckpt_item['upstream'],
            'downstream': ckpt_item['downstream'],
            'p': ckpt_item['p']
        }
        ReceiverModel = FullyConnectedNet(params, layers_dict, ckpt_item['dropout'])
        if args.cuda:
            ReceiverModel = ReceiverModel.cuda()
        ReceiverModel.load_state_dict(torch.load(ckpt_item['checkpoint']))
        print(ckpt_item['checkpoint'] + ' loaded.')

        SNR_array, SER_array, BER_array = test_FCNet(args, 'FCNet', ReceiverModel, testloader)
        if key.find('dropout') > 0:
            SER_results_dropout.update({key: SER_array})
            BER_results_dropout.update({key: BER_array})
        else:
            SER_results.update({key: SER_array})
            BER_results.update({key: BER_array})

    title = 'NT{:d}_NR{:d}_'.format(params['NT'], params['NR']) + params['modulation']
    SER_path = os.path.join(fig_dir, 'FCNet_SER.png')
    plot_comparison(SNR_array, SER_results, 'SER', title, SER_path, SER_results_dropout)
    BER_path = os.path.join(fig_dir, 'FCNet_BER.png')
    plot_comparison(SNR_array, BER_results, 'BER', title, BER_path, BER_results_dropout)
    #print(SER_path + " and " + BER_path + " saved.")


def compare_with_classics(args, fig_dir):
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
        'batch_size': args.batch_size_test,
        'constellation': constellation,
        'cuda': args.cuda
    }

    SNRdB_range_test = np.linspace(args.SNRdB_min, args.SNRdB_max, args.test_size)
    SNRdB_range_test = np.repeat(SNRdB_range_test, args.batch_size_test, axis=0)
    testset = QAM_Dataset(params, SNRdB_range_test)
    testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=2)

    SER_results = {}
    BER_results = {}

    # Test MMNet
    MMNetModel = MMNet(params, MIXED_INFO['MMNet']['linear_name'], MIXED_INFO['MMNet']['denoiser_name'],
                    MIXED_INFO['MMNet']['num_layers'])
    if args.cuda:
        MMNetModel = MMNetModel.cuda()
    MMNetModel.load_state_dict(torch.load(MIXED_INFO['MMNet']['checkpoint']))
    print(MIXED_INFO['MMNet']['checkpoint'] + " loaded.")
    SNR_array, SER_array, BER_array = test_MMNet(args, 'MMNet', MMNetModel, testloader)
    SER_results.update({'MMNet-{:d}layers'.format(MIXED_INFO['MMNet']['num_layers']) : SER_array})
    BER_results.update({'MMNet-{:d}layers'.format(MIXED_INFO['MMNet']['num_layers']) : BER_array})

    # Test FCNet
    layers_dict = {
        'upstream': MIXED_INFO['FCNet']['upstream'],
        'downstream': MIXED_INFO['FCNet']['downstream'],
        'p': MIXED_INFO['FCNet']['p']
    }
    FCNetModel = FullyConnectedNet(params, layers_dict, MIXED_INFO['FCNet']['dropout'])
    if args.cuda:
        FCNetModel = FCNetModel.cuda()
    FCNetModel.load_state_dict(torch.load(MIXED_INFO['FCNet']['checkpoint']))
    print(MIXED_INFO['FCNet']['checkpoint'] + ' loaded.')
    SNR_array, SER_array, BER_array = test_FCNet(args, 'FCNet', FCNetModel, testloader)
    key = 'FCNet-{:d}layers'.format(layers_dict['upstream']+1+layers_dict['downstream'])
    if MIXED_INFO['FCNet']['dropout']:
        key += '-dropout'
    SER_results.update({key: SER_array})
    BER_results.update({key: BER_array})
    
    # Test zero-forcing
    SNR_array, SER_array, BER_array = test_classic(args, 'zero-forcing', testloader)
    SER_results.update({'zero-forcing': SER_array})
    BER_results.update({'zero-forcing': BER_array})

    # Test MMSE
    SNR_array, SER_array, BER_array = test_classic(args, 'MMSE', testloader)
    SER_results.update({'MMSE': SER_array})
    BER_results.update({'MMSE': BER_array})

    # plot figure
    title = 'NT{:d}_NR{:d}_'.format(params['NT'], params['NR']) + params['modulation']
    SER_path = os.path.join(fig_dir, 'compare_with_classics_SER_FC1.png')
    plot_comparison(SNR_array, SER_results, 'SER', title, SER_path)
    BER_path = os.path.join(fig_dir, 'compare_with_classics_BER_FC1.png')
    plot_comparison(SNR_array, BER_results, 'BER', title, BER_path)
    #print(SER_path + " and " + BER_path + " saved.")


if __name__ == '__main__':
    args = parse_args()
    fig_dir = './experiments/comparison_LDLN_SISO'
    compare_MMNet(args, fig_dir)
    #compare_FCNet(args, fig_dir)
    compare_with_classics(args, fig_dir)