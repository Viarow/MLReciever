import torch
from torch.utils.data import Dataset
from torchvision import datasets
from dataset.generator import QAM_Generator, QAM_Generator_Nonlinear


class QAM_Dataset(Dataset):
    def __init__(self, params, SNRdB_range):
        # params must include NT, NR, modulation (QAM_4, QAM_16, QAM64), channel_type
        # SNRdB_range could a numpy array derived from linspace
        self.generator = QAM_Generator(params)
        self.SNRdB_range = SNRdB_range

    def __len__(self):
        return self.SNRdB_range.shape[0]

    def __getitem__(self, idx):
        SNRdB = self.SNRdB_range[idx]
        indices = self.generator.random_indices()
        x = self.generator.modulate(indices)
        #bit_seq = self.generator.map_to_bits(indices)
        y, H, noise_sigma = self.generator.pass_channel(x, SNRdB)
        data_blob = {
            'indices': indices,
            'x': x.type(torch.FloatTensor),
            'y': y.type(torch.FloatTensor),
            'H': H.type(torch.FloatTensor),
            'noise_sigma': noise_sigma.type(torch.FloatTensor),
            'SNRdB': SNRdB
        }
        return data_blob


class QAM_Dataset_Constant(Dataset):
    def __init__(self, params, SNRdB_range):
        self.SNRdB_range = SNRdB_range
        generator = QAM_Generator(params)
        self.generator = generator
        self.indices = generator.random_indices()
        self.x = generator.modulate(self.indices)

    def __len__(self):
        return self.SNRdB_range.shape[0]

    def __getitem__(self, idx):
        SNRdB = self.SNRdB_range[idx]
        y, H, noise_sigma = self.generator.pass_channel(self.x, SNRdB)
        data_blob = {
            'indices': self.indices,
            'x': self.x.type(torch.FloatTensor),
            'y': y.type(torch.FloatTensor),
            'H': H.type(torch.FloatTensor),
            'noise_sigma': noise_sigma.type(torch.FloatTensor),
            'SNRdB': SNRdB
        }
        return data_blob


class QAM_Dataset_Nonlinear(Dataset):
    def __init__(self, params, SNRdB_range):
        # params must include NT, NR, modulation (QAM_4, QAM_16, QAM64), channel_type
        # SNRdB_range could a numpy array derived from linspace
        self.generator = QAM_Generator_Nonlinear(params)
        self.SNRdB_range = SNRdB_range
        self.satlevel = params['satlevel']

    def __len__(self):
        return self.SNRdB_range.shape[0]

    def __getitem__(self, idx):
        SNRdB = self.SNRdB_range[idx]
        indices = self.generator.random_indices()
        x = self.generator.modulate(indices)
        #bit_seq = self.generator.map_to_bits(indices)
        x_amp = self.generator.pass_amplifier(x, self.satlevel)
        y, H, noise_sigma = self.generator.pass_channel(x, SNRdB)
        data_blob = {
            'indices': indices,
            'x': x.type(torch.FloatTensor),
            'y': y.type(torch.FloatTensor),
            'H': H.type(torch.FloatTensor),
            'noise_sigma': noise_sigma.type(torch.FloatTensor),
            'SNRdB': SNRdB
        }
        return data_blob