import torch
from torch import nn
import torch.nn.functional as F
from network.network_utils import *

class Gaussian_Denoiser(nn.Module):
    def __init__(self, params):
        super(Gaussian_Denoiser, self).__init__()
        # lgst_constel must be a torch tensor here to be convert to float32 type
        lgst_constel = params['constellation'].type(torch.FloatTensor)
        if params['cuda']:
            lgst_constel = lgst_constel.cuda()
        self.lgst_constel = lgst_constel
        self.M = int(lgst_constel.shape[0])
        self.NT = 2*params['NT']
        self.NR = 2*params['NR']
        #self.L = params['L']

    def forward(self, zt, features):
        tau2_t = features['tau2_t']
        arg = torch.reshape(zt, (-1,1)) - self.lgst_constel
        arg = torch.reshape(arg, (-1, self.NT, self.M))
        arg = -1. * torch.square(arg) /(2. * tau2_t)
        arg = torch.reshape(arg, (-1, self.M))
        shatt1 = F.softmax(arg, dim=1)
        shatt1 = torch.matmul(shatt1, torch.reshape(self.lgst_constel, (self.M, 1)))
        shatt1 = torch.reshape(shatt1, (-1, self.NT))
        denoiser_helper = {}
        return shatt1, denoiser_helper


class DetNet_Denoiser(nn.Module):
    def __init__(self, params):
        super(DeNet_Denoiser, self).__init__()
        lgst_constel = params['constellation']
        self.lgst_constel = lgst_constel.type(torch.FloatTensor)
        self.M = int(lgst_constel.shape[0])
        self.NT = 2*params['NT']
        self.NR = 2*params['NR']
        #self.L = params['L']
        self.fc1 = nn.Linear(params['NT'], 8*self.NT)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8*self.NT, self.NT)

    def forward(self, zt, xhatt, rt, features, linear_helper):
        H = features['H']
        shatt1 = self.fc1(zt)
        shatt1 = self.relu(shatt1)
        shatt1 = self.fc2(shatt1)
        denoiser_helper = {}
        return shatt1, denoiser_helper

        
class MMNet_Denoiser(nn.Module):
    def __init__(self, params):
        super(MMNet_Denoiser, self).__init__()
        lgst_constel = params['constellation']
        self.lgst_constel = lgst_constel.type(torch.FloatTensor)
        self.M = int(lgst_constel.shape[0])
        self.NT = 2*params['NT']
        self.NR = 2*params['NR']
        #self.L = params['L']
        self.gaussian = Gaussian_Denoiser(params)
        self.use_cuda = params['cuda']

    def forward(self, zt, xhatt, rt, features, linear_helper):
        H = features['H']
        noise_sigma = features['noise_sigma']
        W_t = linear_helper['W']
        HTH = torch.matmul(torch.transpose(H, -2, -1), H)
        v2_t = torch.sum(torch.square(rt), dim=1, keepdim=True) - self.NR * torch.square(noise_sigma)/2.
        v2_t = torch.div(v2_t, torch.unsqueeze(batch_mat_trace(HTH), dim=1))
        if self.use_cuda:
            epsilon = torch.Tensor([1e-9]).cuda()
            eye = torch.eye(self.NT).repeat(H.shape[0], 1, 1).cuda()
            normal_dist = torch.normal(torch.Tensor([[[1.0]]]).repeat(1, self.NT, 1), torch.Tensor([[[0.1]]]).repeat(1, self.NT, 1)).cuda()
        else:
            epsilon = torch.Tensor([1e-9])
            eye = torch.eye(self.NT).repeat(H.shape[0], 1, 1)
            normal_dist = torch.normal(torch.Tensor([[[1.0]]]).repeat(1, self.NT, 1), torch.Tensor([[[0.1]]]).repeat(1, self.NT, 1))

        v2_t = torch.unsqueeze(torch.maximum(v2_t, epsilon), dim=2)
        C_t = eye - torch.matmul(W_t, H)
        tau2_a = 1/self.NT * torch.reshape(batch_mat_trace(torch.matmul(C_t, torch.transpose(C_t, -2, -1))), (-1, 1, 1)) * v2_t
        tau2_b = torch.square(torch.reshape(noise_sigma, (-1, 1, 1))) / (2.*self.NT) * torch.reshape(batch_mat_trace(torch.matmul(W_t, torch.transpose(W_t, -2, -1))), (-1, 1, 1))
        tau2_t = tau2_a + tau2_b
        shatt1, _ = self.gaussian(zt, {'tau2_t': tau2_t/normal_dist})
        denoiser_helper = {}
        return shatt1, denoiser_helper