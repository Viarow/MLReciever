import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def HighPowerAmplifier(y, satlevel=5.):
    y_abs = torch.abs(y)
    alpha = y_abs / (torch.mean(y_abs) * torch.pow(torch.Tensor([10.]), satlevel/10))
    y_amp_abs = torch.div(y_abs, torch.sqrt(1. + torch.pow(alpha, 2)))
    y_amp = y * torch.div(y_amp_abs, y_abs)

    #y_amp = torch.div(y, 1+torch.pow(y/satlevel, 2))
    
    return y_amp


def plot_amp(path):
    y = torch.linspace(0, 10, 200)
    satlevels = [0., 2., 5., 10.]
    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    for level in satlevels:
        y_amp = HighPowerAmplifier(y, level)
        ax.plot(y.numpy(), y_amp.numpy(), label='satlevel={:.1f}'.format(level))
    leg = ax.legend()
    ax.legend(frameon=True)
    ax.set_ylabel('Output Amplitude')
    ax.set_xlabel('Input Amplitude')
    ax.set_title('Amplifier Effects')
    fig.savefig(path, dpi=fig.dpi)

# if __name__ == '__main__':
#     fig_dir = '../experiments/comparison_NDLA_10I10O'
#     if not os.path.exists(fig_dir):
#         os.makedirs(fig_dir)
#     path = os.path.join(fig_dir, 'display_amp_effects.png')
#     plot_amp(path)
#     print(path + " saved.")


def WienerHammerstein(y, order, coefficients):
    '''
    Generalized Wiener Hammerstein Model
    order = len(coefficients)
    TODO: consider time delay
    '''
    zt = torch.Tensor([0.]).repeat(y.shape)
    for k in range(0, order):
        zt += coefficients[k] * y * torch.pow(torch.sqrt(torch.sum(torch.square(y), dim=0)), k)

    return zt