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
    actual_SNRdB = 20*torch.log10(torch.sum(torch.square(r)) / torch.sum(torch.square(noise)))

    return r, H, N0, actual_SNRdB


def RayleighFading(s, SNRdB, std_real, std_imgn, NR, NT):
    '''
    Rayleigh Fading Channel: real component and imaginary component follows a Gaussian 
                            distribution with zero mean and different variance
    std_real & std_imgn: float, standard deviation
    NR & NT: int
    '''
    std_real_mat = torch.Tensor([[std_real]]).repeat([NR, NT])
    std_imgn_mat = torch.Tensor([[std_imgn]]).repeat([NR, NT])
    Hr = torch.normal(mean=torch.Tensor([[0.]]).repeat([NR, NT]), std=std_real_mat)
    Hi = torch.normal(mean=torch.Tensor([[0.]]).repeat([NR, NT]), std=std_imgn_mat)
    h1= torch.cat((Hr, -1.*Hi), dim=1)
    h2 = torch.cat((Hi, Hr), dim=1)
    H = torch.cat((h1, h2), dim=0)
    r = torch.matmul(H.float(), s.float())

    gamma = 10 ** (SNRdB / 20)
    scale = (8. + 4.*(gamma-1) - 8*np.sqrt(gamma)) / np.square(gamma - 1)
    noise_std = scale * torch.sum(torch.square(r)) / r.shape[0]
    noise = (r + np.sqrt(gamma)*r) / (gamma-1)
    #noise = (r + np.sqrt(gamma)*r) / (gamma-1) * torch.randn(r.shape)
    r = r + noise
    noise_sigma = torch.Tensor([noise_std])
    actual_SNRdB = 20*torch.log10(torch.sum(torch.square(r)) / torch.sum(torch.square(noise)))

    return r, H, noise_sigma, actual_SNRdB


def RicianFading(s, SNRdB, mean_real, std_real, mean_imgn, std_imgn, NR, NT):
    '''
    Rician Fading Channel: real component and imaginary component follows a Gaussian 
                            distribution with different mean and variance
    '''
    mean_real_mat = torch.Tensor([[mean_real]]).repeat([NR, NT])
    mean_imgn_mat = torch.Tensor([[mean_imgn]]).repeat([NR, NT])
    std_real_mat = torch.Tensor([[std_real]]).repeat([NR, NT])
    std_imgn_mat = torch.Tensor([[std_imgn]]).repeat([NR, NT])
    Hr = torch.normal(mean=mean_real_mat, std=std_real_mat)
    Hi = torch.normal(mean=mean_imgn_mat, std=std_imgn_mat)
    h1= torch.cat((Hr, -1.*Hi), dim=1)
    h2 = torch.cat((Hi, Hr), dim=1)
    H = torch.cat((h1, h2), dim=0)

    gamma = 10 ** (SNRdB / 20)
    scale = (8. + 4.*(gamma-1) - 8*np.sqrt(gamma)) / np.square(gamma - 1)
    noise_std = scale * torch.sum(torch.square(r)) / r.shape[0]
    noise = (r + np.sqrt(gamma)*r) / (gamma-1)
    #noise = (r + np.sqrt(gamma)*r) / (gamma-1) * torch.randn(r.shape)
    r = r + noise
    noise_sigma = torch.Tensor([noise_std])
    actual_SNRdB = 20*torch.log10(torch.sum(torch.square(r)) / torch.sum(torch.square(noise)))

    return r, H, noise_sigma, actual_SNRdB


# if __name__ == '__main__':
#     NT = 10
#     NR = 10
#     std_real = 0.5
#     std_imgn = 0.5
#     batch_size = 256
#     batch_data = torch.Tensor([[[1]]]).repeat(batch_size, 2*NT, 1)
#     SNRdB_range = np.linspace(2, 22, 20)
#     for SNR in SNRdB_range:
#         actual_SNRdB_list = []
#         for k in range(0, batch_size):
#             s = batch_data[k]
#             r, H, N, actual_SNRdB = RayleighFading(s, SNR, std_real, std_imgn, NR, NT)
#             #r, H, N, actual_SNRdB = AWGN(s, SNR)
#             actual_SNRdB_list.append(actual_SNRdB)
#         average_SNRdB = sum(actual_SNRdB_list) / batch_size
#         print("Ground Truth: {:.2f} -- Simulated: {:.2f}".format(SNR, average_SNRdB))
    
