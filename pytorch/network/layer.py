import torch
from torch import nn
from network.denoiser import *
from network.linear import *
from network.network_utils import batch_matvec_mul
import sys

class Layer(nn.Module):
    def __init__(self, linear_name, denoiser_name, params):
        super(Layer, self).__init__()
        self.NT = params['NT']
        self.batch_size = params['batch_size']
        self.linear_fun = getattr(sys.modules[__name__], linear_name)(params)
        self.denoiser_fun = getattr(sys.modules[__name__], denoiser_name)(params)
        self.linear_fun.reset_parameters()
        self.use_cuda = params['cuda']

    def forward(self, data_blob):
        x = data_blob['x']   #label in batch
        xhat = data_blob['xhat']    #predicted xhat in each layer
        r = data_blob['r']      #difference between y and H * xhat
        features = {
            'y': data_blob['y'],
            'H': data_blob['H'],
            'noise_sigma': data_blob['noise_sigma']
        }                           #contains y, H, noise_sigma

        z, linear_helper = self.linear_fun(xhat, r, features)
        #features['onsager'] = onsager
        new_xhat, denoiser_helper = self.denoiser_fun(z, xhat, r, features, linear_helper)
        new_r = features['y'] - batch_matvec_mul(features['H'], new_xhat)
        #new_onsager = denoiser_helper['onsager']

        W = linear_helper['W']
        I = torch.eye(2*self.NT).repeat(self.batch_size, 1, 1)
        if self.use_cuda:
            I  = I.cuda()
        e10 = batch_matvec_mul(I - torch.matmul(W, features['H']), x-xhat)
        e11 = batch_matvec_mul(W, features['y']-batch_matvec_mul(features['H'], x))
        helper = {'linear': linear_helper, 
                'denoiser': denoiser_helper, 
                'stat':{'e0':e10, 'e11': e11}}
        # currently helper is not used
        
        data_blob['xhat'] = new_xhat
        data_blob['r'] = new_r
        data_blob['helper'] = helper
        
        return data_blob