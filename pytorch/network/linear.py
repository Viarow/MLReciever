import torch
import numpy
from torch import nn
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.init as init
from network_utils import batch_matvec_mul

class MMNet_linear(nn.Module):
    def __init__(self, params):
        super(MMNet_linear, self).__init__()
        self.NT = 2*params['NT']
        self.NR = 2*params['NR']
        self.batch_size = params['batch_size']
        Wr = torch.Tensor(1, self.NT//2, self.NR//2)
        Wi = torch.Tensor(1, self.NT//2, self.NR//2)
        W_cat = torch.cat(torch.cat((Wr, -Wi), dim=2), torch.cat((Wi, Wr), dim=2), dim=1)
        self.W = Parameter(W_cat.repeat(self.batch_size, 1, 1))

    def reset_parameters(self):
        init.normal_(self.W, mean=0., std=0.01)

    def forward(self, shatt, rt, features):
        H = features['H']
        zt = shatt + batch_matvec_mul(self.W, rt)
        helper = {'W':self.W, 'I_WH':torch.eye(self.NT).repeat(self.batch_size, 1, 1)-torch.matmul(self.W, H)}
        return zt, helper


class MMNet_iid_linear(nn.Module):
    def __init__(self, params):
        super(MMNet_iid_linear, self).__init__()
        self.NT = 2*params['NT']
        self.NR = 2*params['NR']
        self.batch_size = params['batch_size']
        self.register_parameter(name='weight', None)

    def reset_parameters(self):
        pass

    def forward(self, shatt, rt, features):
        H = features['H']
        W = torch.transpose(H, 1, 2)
        # zt = shatt + torch.matmul(W.transpose_(1,2), rt.unsqueeze_(dim=2))
        zt = shatt + batch_matvec_mul(W, rt)
        helper = {'W': W}
        return zt, helper


class DetNet_linear(nn.Module):
    def __init__(self, params):
        super(DetNet_iid_linear, self).__init__()
        self.NT = 2*params['NT']
        self.NR = 2*params['NR']
        self.batch_size = params['batch_size']
        self.register_parameter(name='weight', None)
        self.theta1 = Parameter(torch.Tensor(1, self.NT, self.NT).repeat(self.batch_size, 1, 1))
        #self.theta2 = Parameter(torch.Tensor(1))
        self.theta3 = Parameter(torch.Tensor(1, self.NT))

    def reset_parameter(self):
        init.normal_(self.theta1, mean=0., std=0.001)
        #init.uniform_(self.theta2, 1., 1.)
        init.normal_(self.theta3, mean=0., std=0.001)

    def forward(self, shatt, rt, features):
        H = features['H']
        W = torch.transpose(H, 1, 2)
        zt1 = torch.matmul(self.theta1, W)
        zt2 = features['y'] - batch_matvec_mul(H, shatt)
        zt = shatt + batch_matvec_mul(zt1, zt2) + self.theta3
        helper = {'W': W}
        return zt, helper
