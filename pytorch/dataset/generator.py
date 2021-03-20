import torch
import numpy as np
from channel import AWGN
from mapping import QAM_Mapping

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
        indices_QAM = torch.randint(0, np.sqrt(self.mod_n), 2*self.NT)
        return indices_QAM

    def modulate(self, indices):
        x = torch.gather(self.constellation, dim=0, indices)
        return x

    def map_to_bits(self, indices):
        bit_seq = self.mapping.idx_to_bits(indices)
        return bit_seq

    def pass_channel(self, x, SNRdB):
        # TODO: needs channel selection here
        y, H, noise_sigma = AWGN(x, SNRdB)
        #noise_sigma = torch.pow(torch.Tensor([10.]), (10.*np.log10(self.NT) - SNRdB-10.*np.log10(self.NR))/20.)
        return y, H, noise_sigma

