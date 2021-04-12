import torch
import numpy as np
from dataset.channel import *
from dataset.amplifier import HighPowerAmplifier, WienerHammerstein
from dataset.mapping import QAM_Mapping

"""
About NT and NR:
Consider a single sub-carrier in the uplink case,
NT = #transmitting users
NR = #base stations * #receiving antennas per station
"""

class QAM_Generator(object):
    def __init__(self, params):
        # params must include NT, NR, modulation (QAM only), channel type
        self.NT = params['NT']
        self.NR = params['NR']
        self.channel_type = params['channel']
        modulation = params['modulation']
        self.mod_n = int(modulation.split('_')[1])
        self.constellation = self.QAM_N_const()
        self.mapping = QAM_Mapping(modulation)


    def QAM_N_const(self):
        n = self.mod_n
        constellation = np.linspace(-np.sqrt(n)+1, np.sqrt(n)-1, int(np.sqrt(n)))
        alpha = np.sqrt((constellation ** 2).mean())
        constellation /= (alpha * np.sqrt(2))
        constellation = torch.from_numpy(constellation)
        return constellation


    def random_indices(self):
        indices_QAM = torch.randint(low=0, high=int(np.sqrt(self.mod_n)), size=(2*self.NT, ))
        return indices_QAM

    def modulate(self, indices):
        x = torch.gather(self.constellation, dim=0, index=indices)
        return x

    def map_to_bits(self, indices):
        bit_seq = self.mapping.idx_to_bits(indices)
        return bit_seq

    def pass_channel(self, x, SNRdB):
        # TODO: needs channel selection here
        if self.channel_type == 'AWGN':
            y, H, noise_sigma, actual_SNRdB = AWGN(x, SNRdB)
        elif self.channel_type == 'RayleighFading':
            y, H, noise_sigma, actual_SNRdB = RayleighFading(x, SNRdB, 0.5, 0.5, self.NR, self.NT)
        elif self.channel_type == 'RicianFading':
            y, H, noise_sigma, actual_SNRdB = RicianFading(x, SNRdB, 1., 0.5, 1., 0.5, self.NR, self.NT)
        #noise_sigma = torch.pow(torch.Tensor([10.]), (10.*np.log10(self.NT) - SNRdB-10.*np.log10(self.NR))/20.)
        return y, H, noise_sigma, actual_SNRdB


class QAM_Generator_Nonlinear(object):
    def __init__(self, params):
        # params must include NT, NR, modulation (QAM only), channel type
        self.NT = params['NT']
        self.NR = params['NR']
        self.channel_type = params['channel']
        self.amplifier = params['amplifier']
        modulation = params['modulation']
        self.mod_n = int(modulation.split('_')[1])
        self.constellation = self.QAM_N_const()
        self.mapping = QAM_Mapping(modulation)


    def QAM_N_const(self):
        n = self.mod_n
        constellation = np.linspace(-np.sqrt(n)+1, np.sqrt(n)-1, int(np.sqrt(n)))
        alpha = np.sqrt((constellation ** 2).mean())
        constellation /= (alpha * np.sqrt(2))
        constellation = torch.from_numpy(constellation)
        return constellation


    def random_indices(self):
        indices_QAM = torch.randint(low=0, high=int(np.sqrt(self.mod_n)), size=(2*self.NT, ))
        return indices_QAM

    def modulate(self, indices):
        x = torch.gather(self.constellation, dim=0, index=indices)
        return x

    def map_to_bits(self, indices):
        bit_seq = self.mapping.idx_to_bits(indices)
        return bit_seq

    def pass_amplifier(self, x, order, coefficients):
        # TODO: needs amplidier selection here
        if self.amplifier == 'None':
            x_amp = x
        elif self.amplifier == 'WienerHammerstein':
            x_amp = WienerHammerstein(x, order, coefficients)
        return x_amp

    def pass_channel(self, x, SNRdB):
        # TODO: needs channel selection here
        if self.channel_type == 'AWGN':
            y, H, noise_sigma, actual_SNRdB = AWGN(x, SNRdB)
        elif self.channel_type == 'RayleighFading':
            y, H, noise_sigma, actual_SNRdB = RayleighFading(x, SNRdB, 0.5, 0.5, self.NR, self.NT)
        elif self.channel_type == 'RicianFading':
            y, H, noise_sigma, actual_SNRdB = RicianFading(x, SNRdB, 1., 0.5, 1., 0.5, self.NR, self.NT)
        #noise_sigma = torch.pow(torch.Tensor([10.]), (10.*np.log10(self.NT) - SNRdB-10.*np.log10(self.NR))/20.)
        return y, H, noise_sigma, actual_SNRdB