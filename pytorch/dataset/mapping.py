import torch
import numpy as np


class QAM_Mapping(object):
    def __init__(self, modulation):
        """ modulation should be chosen from QAM_4, QAM_16, QAM_64 """
        QAM_4 = {
            0: "0",
            1: "1"
        }
        QAM_16 = {
            0: "00",
            1: "01",
            2: "10",
            3: "11"
        }
        QAM_64 = {
            0: "000",
            1: "001",
            2: "010",
            3: "011",
            4: "100",
            5: "101",
            6: "110",
            7: "111"
        }
        if modulation == 'QAM_4':
            self.map = QAM_4
        elif modulation == 'QAM_16':
            self.map = QAM_16
        else :
            self.map = QAM_64


    def idx_to_bits(self, indices):
        # indices are torch Tensor with shape [2*NT, 1]
        indices = indices.squeeze_().numpy()
        length = indices.shape[0]
        NT = length//2
        real_idx = indices[0: NT]
        imgn_idx = indices[NT: length]
        symbol_list = []
        for k in range(0, NT):
            real_k = real_idx[k]
            imgn_k = imgn_idx[k]
            symbol_list.append(self.map[real_k] + self.map[imgn_k])
        
        bit_seq = ''.join(symbol_list)
        return bit_seq

        