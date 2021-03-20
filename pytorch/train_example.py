import torch
import torch.optim as optim
import torch.nn as nn
from dataset.simulated_dataset import QAM_Dataset
from network.detector import FullyConnectedNet
from utils import *
from tqdm import tqdm
import numpy as np


def main():
    params = {
        'NT': 1,
        'NR': 1,
        'modulation': 'QAM_16',
        'channel_type': 'AWGN'
    }
    layers_dict = {'upstream':4, 'downstream':4}
    ReceiverModel = FullyConnectedNet(params, layers_dict, dropout=False)

    SNRdB_range_train = np.linspace(5, 25, 6400)
    trainset = QAM_Dataset(params, SNRdB_range_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, num_workers=2, shuffle=False)
    SNRdB_range_test = np.linspace(5, 25, 640)
    testset = QAM_Dataset(params, SNRdB_range_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=1, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(ReceiverModel.parameters(), lr=1e-3, momentum=0.9)
    ReceiverModel = ReceiverModel.cuda()
    num_epochs = 100
    for epoch in range(0, num_epochs):
        running_loss = 0.0
        for i, feature, label in enumerate(trainloader, 0):
            x = label['symbol'].cuda()
            y = feature['y'].cuda()
            #H = feature['H']
            #noise_sigma = features['noise_sigma']
            #SNRdB = feature['SNRdB']
            optimizer.zero_grad()
            xhat = ReceiverModel(y)
            loss = criterion(xhat, x)
            optimizer.step()
            running_loss += loss.item()
            


    