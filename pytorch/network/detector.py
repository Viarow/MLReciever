import torch
from torch import nn
from network.layer import Layer


class MMNet(nn.Module):
    def __init__(self, params, linear_name='MMNet_linear', denoiser_name='MMNet_Denoiser', num_layers=10):
        super(MMNet, self).__init__()
        # params must include: NT, NR, constellation, batch_size
        self.params = params
        self.NT = 2*params['NT']
        self.NR = 2*params['NR']
        self.batch_size = params['batch_size']
        self.num_layers = num_layers
        self.use_cuda = params['cuda']
        
        layer_list = []
        for k in range(1, num_layers+1):
            layer_k = Layer(linear_name, denoiser_name, params)
            layer_list.append(layer_k)
        self.layers = nn.Sequential(*layer_list)

    def forward(self, x, y, H, noise_sigma):
        if self.use_cuda:
            xhat = torch.zeros((self.batch_size, self.NT)).type(torch.FloatTensor).cuda()
            r = y.type(torch.FloatTensor).cuda()
        else:
            xhat = torch.zeros((self.batch_size, self.NT)).type(torch.FloatTensor)
            r = y.type(torch.FloatTensor)
        input_blob = {
            'x': x,
            'xhat': xhat,
            'r': r,
            'y': y,
            'H': H,
            'noise_sigma': noise_sigma,
            'helper': {}
        }

        #pred_blob = self.layers(input_blob)
        #return pred_blob['xhat'], pred_blob['helper']
        xhat_list = []
        for k in range(0, self.num_layers):
            pred_blob = self.layers[k](input_blob)
            xhat_list.append(pred_blob['xhat'])
            input_blob = pred_blob

        return xhat_list
        


class FullyConnectedNet(nn.Module):
    def __init__(self, params, layers_dict, dropout=False):
        # params must include: NT, NR, constellation, batch_size
        # layers_dict must include: number of upstream layers and downstream layers, 
        # ......................... dropout rate p(in dropout=false, set as 0)
        # non-linear activation uses PReLU, which has a learnable parameter a.
        super(FullyConnectedNet, self).__init__()
        self.NT = 2*params['NT']
        self.NR = 2*params['NR']
        self.num_layers = layers_dict['upstream'] + 1 + layers_dict['downstream']

        layers_list = []
        if dropout:
            for i in range(0, layers_dict['upstream']):
                layers_list.append(nn.Linear(in_features=self.NR, out_features=self.NR))
                layers_list.append(nn.PReLU())
                layers_list.append(nn.Dropout(p=layers_dict['p']))

            layers_list.append(nn.Linear(in_features=self.NR, out_features=self.NT))
            layers_list.append(nn.PReLU())
            layers_list.append(nn.Dropout(p=layers_dict['p']))

            for j in range(0, layers_dict['downstream']):
                layers_list.append(nn.Linear(in_features=self.NT, out_features=self.NT))
                layers_list.append(nn.PReLU())
                layers_list.append(nn.Dropout(p=layers_dict['p']))

        else:
            for i in range(0, layers_dict['upstream']):
                layers_list.append(nn.Linear(in_features=self.NR, out_features=self.NR))
                layers_list.append(nn.PReLU())
            layers_list.append(nn.Linear(in_features=self.NR, out_features=self.NT))
            layers_list.append(nn.PReLU())
            for j in range(0, layers_dict['downstream']):
                layers_list.append(nn.Linear(in_features=self.NT, out_features=self.NT))
                layers_list.append(nn.PReLU())

        self.layers = nn.Sequential(*layers_list)

    def forward(self, y):

        xhat_list = []
        for k in range(0, self.num_layers):
            xhat = self.layers[k](y)
            xhat_list.append(xhat)
            y = xhat
        #xhat = self.layers(y)
        #return xhat
        return xhat_list