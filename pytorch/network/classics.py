import torch
import numpy
from network_utils import batch_matvec_mul


def zero_forcing(y, H):
    '''
    Inputs: 
    y.shape = [batch_size, N] = [batch_size, 2*NR]
    H.shape = [batch_size, N, K] = [batch_size, 2*NR, 2*NT]
    Outputs:
    x.shape = [batch_size, K] = [batch_size, 2*NT]
    '''
    # Projected channel output
    Hty = batch_matvec_mul(torch.transpose(H, 1, 2), y)

    # Gramian of transposed channel matrix
    HtH = torch.matmul(H.transpose_(1, 2), H)

    # Inverse Gramian
    HtHinv = torch.inverse(HtH)

    #Zero-Forcing Detector
    x = batch_matvec_mul(HtHinv, Hty)

    return x


def MMSE(y, H, noise_sigma):
    '''
    Inputs:
    y.shape = [batch_size, N] = [batch_size, 2*NR]
    H.shape = [batch_size, N, K] = [batch_size, 2*NR, 2*NT]
    noise_sigma.shape = [batch_size]
    Outputs:
    x.shape = [batch_size, K] = [batch_size, 2*NT]
    '''
    # Projected channel output
    Hty = batch_matvec_mul(torch.transpose(H, 1, 2), y)

    # Gramian of transposed channel matrix
    HtH = torch.matmul(H.transpose_(1, 2), H) 

    # Inverse Gramian
    batch_size = H.shape[0]
    noise_sqrt = torch.reshape(torch.sqrt(noise_sigma)/2, (-1, 1, 1)) * torch.eye(H.shape[-1]).repeat(batch_size, 1, 1)
    HtHinv = torch.inverse(HtH + noise_sqrt)

    # MMSE detector
    x = batch_matvec_mul(HtHinv, Hty)

    return x