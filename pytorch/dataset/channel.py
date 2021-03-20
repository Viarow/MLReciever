import torch
import numpy as np


def AWGN(s, SNRdB, L=1):
    """
    Parameters:
    s: input signal Tensor with shape [2*NT, 1]
    SNRdB: desired SNR
    L: oversampling factor (applicable for waveform simulation) default L = 1.
    
    Returns:
    r: received signal vector
    H: channel matrix, an identity matrix with shape [2*NT, 2*NT], while NT equals NR.
    """
    gamma = 10**(SNRdB/10)
    power = L*torch.sum(torch.square(torch.abs(s)), dim=0)/s.shape[0]
    N0 = torch.Tensor([power/gamma]).type(torch.FloatTensor)
    noise = torch.sqrt(N0) * torch.randn(s.shape)
    r = s + noise
    H = torch.eye(s.shape[0])

    return r, H, N0