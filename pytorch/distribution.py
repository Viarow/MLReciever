import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.simulated_dataset import QAM_Dataset, QAM_Dataset_Nonlinear
from dataset.mapping import QAM_Mapping
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    # About Dataset
    parser.add_argument('--BaseStation', type=int, default=1, help='Number of base stations')
    parser.add_argument('--Antenna', type=int, default=1, help='Number of receiving antennas per base stattion')
    parser.add_argument('--User', type=int, default=1, help='Number of transmitting users')
    parser.add_argument('--modulation', type=str, default='QAM_16', help='Modulation scheme')
    parser.add_argument('--channel', type=str, default='AWGN', help='Channel Type')
    parser.add_argument('--amplifier', type=str, default='WienerHammerstein', help='Amplifier type')
    parser.add_argument('--SNRdB_min', type=float, default=5, help='Minimum SNR expressed in dB')
    parser.add_argument('--SNRdB_max', type=float, default=5, help='Maximum SNR expressed in dB')
    parser.add_argument('--train_size', type=int, default=6400, help='Size of traing dataset')
    parser.add_argument('--batch_size_train', type=int, default=64, help='Training batch size')
    parser.add_argument('--test_size', type=int, default=20, help="Size of testing dataset")
    parser.add_argument('--batch_size_test', type=int, default=100, help="Test batch size to compute error.")
    parser.add_argument('--fig_dir', type=str, help='The folder to store logging file, model state dict and figures')
    args = parser.parse_args()
    return args


def get_counting_dict(args, dataloader):
    counting_dict = {}
    mapping = QAM_Mapping(args.modulation)
    bit_map = mapping.map
    map_keys = list(bit_map.keys())
    for imag_idx in map_keys:
        for real_idx in map_keys:
            counting_dict.update({bit_map[imag_idx]+bit_map[real_idx] : 0})
    
    for i, data_blob in tqdm(enumerate(dataloader, 0)):
        # batch_size = 0
        indices = data_blob['indices'][0]
        bit_seq = mapping.idx_to_bits(indices)
        counting_dict[bit_seq] += 1

    return counting_dict


def collect_indices(args, params):

    SNRdB_range_train = np.linspace(args.SNRdB_min, args.SNRdB_max, args.train_size)
    trainset = QAM_Dataset(params, args.train_size, SNRdB_range_train)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    SNRdB_range_test = np.linspace(args.SNRdB_min, args.SNRdB_max, args.test_size)
    SNRdB_range_test = np.repeat(SNRdB_range_test, args.batch_size_test, axis=0)
    testset = QAM_Dataset(params, (args.test_size*args.batch_size_test), SNRdB_range_test)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    counting_dict_train = get_counting_dict(args, trainloader)
    counting_dict_test = get_counting_dict(args, testloader)

    return counting_dict_train, counting_dict_test


def plot_distribution(counting_dict, title, path):

    labels = list(counting_dict.keys())
    times = [counting_dict[k] for k in labels]
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(range(len(labels)), times, tick_label=labels)
    ax.set_title(title)
    fig.savefig(path)



if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.fig_dir):
        os.makedirs(args.fig_dir)
    params = {
        'NR': args.BaseStation * args.Antenna,
        'NT': args.User,
        'modulation': args.modulation,
        'channel': args.channel,
        'batch_size': args.batch_size_train
    }
    counting_dict_train, counting_dict_test = collect_indices(args, params)

    title = args.modulation + ' trainset distribution'
    path = os.path.join(args.fig_dir, 'trainset_distribution_balanced_' + args.modulation + '.png')
    plot_distribution(counting_dict_train, title, path)

    title = args.modulation + ' testset distribution'
    path = os.path.join(args.fig_dir, 'testset_distribution_balanced_' + args.modulation + '.png')
    plot_distribution(counting_dict_test, title, path)