import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset.mapping import QAM_Mapping
import os


def get_QAMconstellation(n):
    # Returns a torch Tensor in shape of [sqrt(n), 1]
    constellation = np.linspace(-np.sqrt(n)+1, np.sqrt(n)-1, int(np.sqrt(n)))
    alpha = np.sqrt((constellation ** 2).mean())
    constellation /= (alpha * np.sqrt(2))
    constellation = torch.from_numpy(constellation)
    return constellation


def QAM_demodulate(y, constellation):
    shape = y.shape
    y = torch.reshape(y, (-1, 1))
    constellation = torch.reshape(constellation, (1, -1))
    indices = torch.argmin(torch.abs(y - constellation), dim=1).type(torch.IntTensor)
    indices = torch.reshape(indices, shape)
    return indices


def symbol_accuracy(x, y):
    acc = torch.mean(torch.eq(x, y).type(torch.FloatTensor))
    return float(acc.numpy())


def bit_accuracy(x, y):
    # x, y are two strings
    length = len(x)
    counter = 0
    for k in range(0, length):
        if x[k] == y[k]:
            counter += 1

    return float(counter/length)


def batch_symbol_acc(x_batch, y_batch):
    # x_batch: indices x, y_batch: indices xhat
    batch_size = x_batch.shape[0]
    acc = 0. 
    for k in range(0, batch_size):
        acc += symbol_accuracy(x_batch[k], y_batch[k])
    
    return float(acc/batch_size)


def batch_bit_acc(args, x_batch, y_batch):
    # x_batch: indices x, y_batch: indices xhat
    batch_size = x_batch.shape[0]
    mapping = QAM_Mapping(args.modulation)
    acc = 0.
    for k in range(0, batch_size):
        x_bitseq = mapping.idx_to_bits(x_batch[k])
        y_bitseq = mapping.idx_to_bits(y_batch[k])
        acc += bit_accuracy(x_bitseq, y_bitseq)

    return float(acc/batch_size)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_epochs_FCNet(params, args, SNRdB_range, error_list):
    # Comparing between different epochs for a specific architecture
    error_list = [error_list[-1]]
    num_plots = len(error_list)

    fig_s, ax_s = plt.subplots()
    for k in range(0, num_plots):
        ax_s.plot(SNRdB_range, error_list[k]['SER'], alpha=(0.4+0.6/num_plots*k), label=str(error_list[k]['epoch']))
    ax_s.set_yscale('log')
    leg_s = ax_s.legend()
    ax_s.legend(loc = 'upper right', frameon=True)
    ax_s.set_xlabel('SNR(dB)')
    ax_s.set_ylabel('SER')
    title_s = '{:d}I{:d}O '.format(params['NT'], params['NR']) + params['modulation'] + ' ' 
    title_s += '{:d}layers'.format(args.upstream + args.downstream)
    ax_s.set_title(title_s)
    plt.grid()
    if args.dropout:
        path_s = 'SER_upstream{:d}_downstream{:d}_dropout.png'
    else:
        path_s = 'SER_upstream{:d}_downstream{:d}.png'
    path_s = os.path.join(args.log_dir, path_s.format(args.upstream, args.downstream))
    fig_s.savefig(path_s, dpi=fig_s.dpi)

    fig_b, ax_b = plt.subplots()
    for k in range(0, num_plots):
        ax_b.plot(SNRdB_range, error_list[k]['BER'], alpha=(0.4+0.6/num_plots*k), label=str(error_list[k]['epoch']))
    ax_b.set_yscale('log')
    leg_b = ax_b.legend()
    ax_b.legend(loc = 'upper right', frameon=True)
    ax_b.set_xlabel('SNR(dB)')
    ax_b.set_ylabel('BER')
    title_b = '{:d}I{:d}O '.format(params['NT'], params['NR']) + params['modulation'] + ' '
    title_b += '{:d}layers'.format(args.upstream + args.downstream)
    ax_b.set_title(title_b)
    plt.grid()
    if args.dropout:
        path_b = 'BER_upstream{:d}_downstream{:d}_dropout.png'
    else:
        path_b = 'BER_upstream{:d}_downstream{:d}.png'
    path_b = os.path.join(args.log_dir, path_b.format(args.upstream, args.downstream))
    fig_b.savefig(path_b, dpi=fig_b.dpi)

    return path_s, path_b


def plot_loss(params, args, network, iterations, losses):
    fig, ax = plt.subplots()
    ax.plot(iterations, losses)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    title = '{:d}I{:d}O '.format(params['NT'], params['NR']) + params['modulation'] + ' '
    if network == 'FCNet':
        title += '{:d}layers'.format(args.upstream + args.downstream)
        path = os.path.join(args.log_dir, "loss_FCNet_{:d}upstream_{:d}downstream.png".format(args.upstream, args.downstream))
    elif network == 'MMNet':
        title += '{:d}layers'.format(args.num_layers)
        path = os.path.join(args.log_dir, "loss_MMNet_{:d}layers.png".format(args.num_layers))
    ax.set_title(title)
    plt.grid()
    fig.savefig(path, dpi=fig.dpi)

    return path


def plot_epochs_MMNet(params, args, SNRdB_range, error_list):
    # Comparing between different epochs for a specific architecture
    error_list = [error_list[-1]]
    num_plots = len(error_list)

    fig_s, ax_s = plt.subplots()
    for k in range(0, num_plots):
        ax_s.plot(SNRdB_range, error_list[k]['SER'], alpha=(0.4+0.6/num_plots*k), label=str(error_list[k]['epoch']))
    ax_s.set_yscale('log')
    leg_s = ax_s.legend()
    ax_s.legend(loc = 'upper right', frameon=True)
    ax_s.set_xlabel('SNR(dB)')
    ax_s.set_ylabel('SER')
    title_s = '{:d}I{:d}O '.format(params['NT'], params['NR']) + params['modulation'] + ' ' 
    title_s += '{:d}layers'.format(args.num_layers)
    ax_s.set_title(title_s)
    plt.grid()
    path_s = "SER_MMNet_{:d}layers.png".format(args.num_layers)
    path_s = os.path.join(args.log_dir, path_s)
    fig_s.savefig(path_s, dpi=fig_s.dpi)

    fig_b, ax_b = plt.subplots()
    for k in range(0, num_plots):
        ax_b.plot(SNRdB_range, error_list[k]['BER'], alpha=(0.4+0.6/num_plots*k), label=str(error_list[k]['epoch']))
    ax_b.set_yscale('log')
    leg_b = ax_b.legend()
    ax_b.legend(loc = 'upper right', frameon=True)
    ax_b.set_xlabel('SNR(dB)')
    ax_b.set_ylabel('BER')
    title_b = '{:d}I{:d}O '.format(params['NT'], params['NR']) + params['modulation'] + ' '
    title_b += '{:d}layers'.format(args.num_layers)
    ax_b.set_title(title_b)
    plt.grid()
    path_b = "BER_MMNet_{:d}layers.png".format(args.num_layers)
    path_b = os.path.join(args.log_dir, path_b)
    fig_b.savefig(path_b, dpi=fig_b.dpi)

    return path_s, path_b