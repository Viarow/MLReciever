import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.simulated_dataset import QAM_Dataset, QAM_Dataset_Nonlinear
from dataset.mapping import QAM_Mapping
from network.detector import FullyConnectedNet, MMNet
from network.classics import zero_forcing, MMSE
from utils import *
import numpy as np
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
    parser.add_argument('--amplifier', type=str, default='None', help='Amplifier Type')
    parser.add_argument('--satlevel', type=float, default=0.5, help='Saturation level of high power amplifier')
    parser.add_argument('--SNRdB_min', type=float, default=5, help='Minimum SNR expressed in dB')
    parser.add_argument('--SNRdB_max', type=float, default=5, help='Maximum SNR expressed in dB')
    parser.add_argument('--test_size', type=int, default=20, help="Size of testing dataset")
    parser.add_argument('--batch_size_test', type=int, default=100, help="Test batch size to compute error.")
    parser.add_argument('--cuda', type=bool, default=True, help='Set true when cuda is available')
    args = parser.parse_args()
    return args


MIXED_INFO = {
    'MMNet': {
        'linear_name':'MMNet_linear',
        'denoiser_name': 'MMNet_Denoiser',
        'num_layers': 10,
        'checkpoint': 'experiments_AWGN/10I10O_QAM16_AWGN_LINEAR_MMNet_500epochs/MMNet_10layers_epoch500.pth'
    },
    'FCNet': {
        'dropout': False,
        'upstream': 0,
        'downstream': 0,
        'p': 0.,
        'checkpoint': 'experiments_AWGN/10I10O_QAM16_AWGN_LINEAR_FCNet_500epochs/upstream0_downstream0_epoch500.pth'
    }
}

def display_amp_effects(args, path):

    params = {
        'NR': args.BaseStation * args.Antenna,
        'NT': args.User,
        'modulation': args.modulation,
        'channel': args.channel,
    }

    SNRdB_range_test = np.linspace(args.SNRdB_min, args.SNRdB_max, args.test_size)
    SNRdB_range_test = np.repeat(SNRdB_range_test, args.batch_size_test, axis=0)
    testset_linear = QAM_Dataset(params, SNRdB_range_test)
    testloader_linear = DataLoader(testset_linear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)
    # testset_nonlinear = QAM_Dataset_Nonlinear(params, SNRdB_range_test)
    # testloader_nonlinear = DataLoader(testset_nonlinear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)

    power_linear = []
    SNR_list = []
    for i, data_blob in enumerate(testloader_linear, 0):
        y = data_blob['y']
        batch_power = torch.mean(torch.pow(y, 2), dim=1)
        power_linear.append(torch.mean(batch_power, dim=0).squeeze().item())
        SNRdB = data_blob['SNRdB']
        SNR_list.append(SNRdB[0])

    SNR_array = np.asarray(SNR_list)
    power_dict = {'no amplifier': np.asarray(power_linear)}
    params.update({'amplifier': args.amplifier, 'order':1, 'coefficients': [1.0]})

    for order in range(1, 5):
        params['order'] = order
        if order > 1:
            params['coefficients'].append(1.0/(2**(order+2)))
        testset_nonlinear = QAM_Dataset_Nonlinear(params, SNRdB_range_test)
        testloader_nonlinear = DataLoader(testset_nonlinear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)
        power_nonlinear = []
        for i, data_blob in enumerate(testloader_nonlinear, 0):
            y = data_blob['y']
            batch_power = torch.mean(torch.square(y), dim=1)
            power_nonlinear.append(torch.mean(batch_power, dim=0).squeeze().item())
        power_dict.update({'order={:d}'.format(order) : np.asarray(power_nonlinear)})

    title = 'NT{:d}_NR{:d}_'.format(params['NT'], params['NR']) + params['modulation']
    plot_comparison(SNR_array, power_dict, 'averaged power', title, path)


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


def compare_MMNet(args, satlevels, fig_dir):
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
        'satlevel': args.satlevel,
        'batch_size': args.batch_size_test,
        'constellation': constellation,
        'cuda': args.cuda
    }
    
    MMNet_info = MIXED_INFO['MMNet']
    ReceiverModel = MMNet(params, MMNet_info['linear_name'], MMNet_info['denoiser_name'],
                    MMNet_info['num_layers'])
    if args.cuda:
        ReceiverModel = ReceiverModel.cuda()
    ReceiverModel.load_state_dict(torch.load(MMNet_info['checkpoint']))
    print(MMNet_info['checkpoint'] + " loaded.")

    SNRdB_range_test = np.linspace(args.SNRdB_min, args.SNRdB_max, args.test_size)
    SNRdB_range_test = np.repeat(SNRdB_range_test, args.batch_size_test, axis=0)

    testset_linear = QAM_Dataset(params, SNRdB_range_test)
    testloader_linear = DataLoader(testset_linear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)
    SNR_array, SER_array, BER_array = test_MMNet(args, 'MMNet', ReceiverModel, testloader_linear)
    SER_results = {'no amplifier' : SER_array}
    BER_results = {'no amplifier' : BER_array}

    for level in satlevels:
        params['satlevel'] = level
        testset_nonlinear = QAM_Dataset_Nonlinear(params, SNRdB_range_test)
        testloader_nonlinear = DataLoader(testset_nonlinear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)
        SNR_array, SER_array, BER_array = test_MMNet(args, 'MMNet', ReceiverModel, testloader_nonlinear)
        SER_results.update({'satlevel={:.1f}'.format(level) : SER_array})
        BER_results.update({'satlevel={:.1f}'.format(level) : BER_array})

    title = 'NT{:d}_NR{:d}_'.format(params['NT'], params['NR']) + params['modulation']
    SER_path = os.path.join(fig_dir, 'SER_MMNet_satlevels.png')
    plot_comparison(SNR_array, SER_results, 'SER', title, SER_path)
    BER_path = os.path.join(fig_dir, 'BER_MMNet_satlevels.png')
    plot_comparison(SNR_array, BER_results, 'BER', title, BER_path)


def compare_FCNet(args, satlevels, fig_dir):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    mod_n = int(args.modulation.split('_')[1])

    params = {
        'NR': args.BaseStation * args.Antenna,
        'NT': args.User,
        'modulation': args.modulation,
        'channel': args.channel,
        'amplifier': args.amplifier,
        'satlevel': args.satlevel
    }

    FCNet_info = MIXED_INFO['FCNet']
    layers_dict = {
        'upstream': FCNet_info['upstream'],
        'downstream': FCNet_info['downstream'],
        'p': FCNet_info['p']
    }
    ReceiverModel = FullyConnectedNet(params, layers_dict, FCNet_info['dropout'])
    if args.cuda:
        ReceiverModel = ReceiverModel.cuda()
    ReceiverModel.load_state_dict(torch.load(FCNet_info['checkpoint']))
    print(FCNet_info['checkpoint'] + " loaded.")

    SNRdB_range_test = np.linspace(args.SNRdB_min, args.SNRdB_max, args.test_size)
    SNRdB_range_test = np.repeat(SNRdB_range_test, args.batch_size_test, axis=0)

    testset_linear = QAM_Dataset(params, SNRdB_range_test)
    testloader_linear = DataLoader(testset_linear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)
    SNR_array, SER_array, BER_array = test_FCNet(args, 'FCNet', ReceiverModel, testloader_linear)
    SER_results = {'no amplifier' : SER_array}
    BER_results = {'no amplifier' : BER_array}

    for level in satlevels:
        params['satlevel'] = level
        testset_nonlinear = QAM_Dataset_Nonlinear(params, SNRdB_range_test)
        testloader_nonlinear = DataLoader(testset_nonlinear, batch_size=args.batch_size_test, shuffle=False, num_workers=2)
        SNR_array, SER_array, BER_array = test_FCNet(args, 'FCNet', ReceiverModel, testloader_nonlinear)
        SER_results.update({'satlevel={:.1f}'.format(level) : SER_array})
        BER_results.update({'satlevel={:.1f}'.format(level) : BER_array})

    title = 'NT{:d}_NR{:d}_'.format(params['NT'], params['NR']) + params['modulation']
    SER_path = os.path.join(fig_dir, 'SER_FCNet_satlevels.png')
    plot_comparison(SNR_array, SER_results, 'SER', title, SER_path)
    BER_path = os.path.join(fig_dir, 'BER_FCNet_satlevels.png')
    plot_comparison(SNR_array, BER_results, 'BER', title, BER_path)


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
        'amplifier': args.amplifier,
        'order': 4,
        'coefficients': [1.0],
        'batch_size': args.batch_size_test,
        'constellation': constellation,
        'cuda': args.cuda
    }

    for order in range(1, 5):
        params['order'] = order
        if order > 1:
            params['coefficients'].append(1.0/(2**(order+2)))


    SER_results = {}
    BER_results = {}

    # linear dataset
    SNRdB_range_test = np.linspace(args.SNRdB_min, args.SNRdB_max, args.test_size)
    SNRdB_range_test = np.repeat(SNRdB_range_test, args.batch_size_test, axis=0)
    testset = QAM_Dataset(params, SNRdB_range_test)
    testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=2)

    # MMNet on linear dataset
    MMNetModel = MMNet(params, MIXED_INFO['MMNet']['linear_name'], MIXED_INFO['MMNet']['denoiser_name'],
                    MIXED_INFO['MMNet']['num_layers'])
    if args.cuda:
        MMNetModel = MMNetModel.cuda()
    MMNetModel.load_state_dict(torch.load(MIXED_INFO['MMNet']['checkpoint']))
    print(MIXED_INFO['MMNet']['checkpoint'] + " loaded.")
    SNR_array, SER_array, BER_array = test_MMNet(args, 'MMNet', MMNetModel, testloader)
    SER_results.update({'MMNet-{:d}layers'.format(MIXED_INFO['MMNet']['num_layers']) : SER_array})
    BER_results.update({'MMNet-{:d}layers'.format(MIXED_INFO['MMNet']['num_layers']) : BER_array})

    # FCNet on linear dataset
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

    # zero-forcing on linear dataset
    SNR_array, SER_array, BER_array = test_classic(args, 'zero-forcing', testloader)
    SER_results.update({'zero-forcing': SER_array})
    BER_results.update({'zero-forcing': BER_array})

    # MMSE on linear dataset
    SNR_array, SER_array, BER_array = test_classic(args, 'MMSE', testloader)
    SER_results.update({'MMSE': SER_array})
    BER_results.update({'MMSE': BER_array})

    SER_results_extra = {}
    BER_results_extra = {}

    # non-linar dataset
    testset = QAM_Dataset_Nonlinear(params, SNRdB_range_test)
    testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=2)

    # MMNet on non-linear dataset
    SNR_array, SER_array, BER_array = test_MMNet(args, 'MMNet', MMNetModel, testloader)
    SER_results_extra.update({'MMNet-{:d}layers(amp)'.format(MIXED_INFO['MMNet']['num_layers']) : SER_array})
    BER_results_extra.update({'MMNet-{:d}layers(amp)'.format(MIXED_INFO['MMNet']['num_layers']) : BER_array})

    # FCNet on non-linear dataset
    SNR_array, SER_array, BER_array = test_FCNet(args, 'FCNet', FCNetModel, testloader)
    key = 'FCNet-{:d}layers(amp)'.format(layers_dict['upstream']+1+layers_dict['downstream'])
    if MIXED_INFO['FCNet']['dropout']:
        key += '-dropout'
    SER_results_extra.update({key: SER_array})
    BER_results_extra.update({key: BER_array})

    # zero-forcing on non-linear dataset
    SNR_array, SER_array, BER_array = test_classic(args, 'zero-forcing', testloader)
    SER_results_extra.update({'zero-forcing(amp)': SER_array})
    BER_results_extra.update({'zero-forcing(amp)': BER_array})

    # MMSE on linear dataset
    SNR_array, SER_array, BER_array = test_classic(args, 'MMSE', testloader)
    SER_results_extra.update({'MMSE(amp)': SER_array})
    BER_results_extra.update({'MMSE(amp)': BER_array})

    # plot figure
    title = 'NT{:d}_NR{:d}_'.format(params['NT'], params['NR']) + params['modulation']
    SER_path = os.path.join(fig_dir, 'compare_with_classics_SER.png')
    plot_comparison(SNR_array, SER_results, 'SER', title, SER_path, SER_results_extra)
    BER_path = os.path.join(fig_dir, 'compare_with_classics_BER.png')
    plot_comparison(SNR_array, BER_results, 'BER', title, BER_path, BER_results_extra)


if __name__ == '__main__':
    args = parse_args()
    fig_dir = './experiments_compare/comparison_NDLA_MIMO'
    compare_with_classics(args, fig_dir)
    #display_amp_effects(args, fig_dir)