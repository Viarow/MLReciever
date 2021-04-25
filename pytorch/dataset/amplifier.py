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


def WienerHammerstein(y, order, coefficients):
    '''
    Generalized Wiener Hammerstein Model
    order = len(coefficients)
    y should be complex float tensor (torch.cfloat)
    coefficients can be real or complex
    TODO: consider time delay
    '''
    zt = torch.Tensor([0.]).repeat(y.shape)
    zt = zt.type(torch.cfloat)
    for k in range(0, order):
        zt += coefficients[k] * y * torch.pow(y.abs(), k)

    return zt


def get_complex_QAMconstellation(mod_n):
    n = mod_n
    constellation = np.linspace(-np.sqrt(n)+1, np.sqrt(n)-1, int(np.sqrt(n)))
    alpha = np.sqrt((constellation ** 2).mean())
    constellation /= (alpha * np.sqrt(2))
    constellation = torch.from_numpy(constellation)
    
    idx = 0
    container = torch.randn(mod_n, 2)
    for i in range(0, constellation.shape[0]):
        for j in range(0, constellation.shape[0]):
            container[idx][0] = constellation[i]
            container[idx][1] = constellation[j]
            idx += 1
    complex_constellation = torch.view_as_complex(container)
    
    return complex_constellation


def signal_power(s):
    # s is a torch tensor with dtype=cfloat
    power = torch.mean(torch.square(s.abs()), dim=0)
    return power


def PA_effects_function(fig_dir):

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    orders = [1, 2, 3]
    coefficients = {
        1: [1.0],
        2: [1.0, -0.01],
        3: [1.0, -0.01, -0.0001]
    }
    y = torch.linspace(0, 40, 2000, dtype=torch.cfloat)

    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    for k in orders:
        z = WienerHammerstein(y, k, coefficients[k])
        ax.plot(y.real, z.real, label='K={:d}'.format(k))
    leg = ax.legend()
    ax.legend(frameon=True)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title("W-H Power Amplifier's Effects on Real Input")
    path = os.path.join(fig_dir, 'function.png')
    fig.savefig(path, dpi=fig.dpi)


def PA_effects_constellation(mod_n, fig_dir):

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    constellation = get_complex_QAMconstellation(mod_n)
    constellation_1 = WienerHammerstein(constellation, order=2, coefficients=[1.0, -0.01])
    constellation_2 = WienerHammerstein(constellation, order=2, coefficients=[1.0, -0.1])

    sns.set_style('whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5), constrained_layout=True)
    ax1.scatter(constellation.real, constellation.imag, alpha=0.5, s=40, label='original')
    ax1.scatter(constellation_1.real, constellation_1.imag, alpha=0.5, s=40, label='PA [1.0, -0.01]')
    leg1 = ax1.legend()
    ax1.legend(frameon=True)
    ax1.set_xlabel('real')
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylabel('imag')
    ax1.set_ylim(-1.5, 1.5)

    ax2.scatter(constellation.real, constellation.imag, alpha=0.5, s=40, label='original')
    ax2.scatter(constellation_2.real, constellation_2.imag, alpha=0.5, s=40, label='[1.0, -0.1]')
    leg2 = ax2.legend()
    ax2.legend(frameon=True)
    ax2.set_xlabel('real')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylabel('imag')
    ax2.set_ylim(-1.5, 1.5)

    #fig.tight_layout()
    fig.suptitle("2-order W-H Power Amplifier's Effects on constellation")
    path = os.path.join(fig_dir, 'constellation.png')
    fig.savefig(path, dpi=fig.dpi)


# if __name__ == '__main__':
#     fig_dir = './PA_Evaluation'
#     #PA_effects_function(fig_dir)
#     PA_effects_constellation(256, fig_dir)