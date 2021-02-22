from sample_generator import produce_simulation
from numpy as np
import os

NT = 16
NR = 64
params_list = []
snr_min = 11.0
snr_max = 17.0
intervals = 8

for snr_k in np.linspace(snr_min, snr_max, intervals+1)[:-1]:
    params = {
        'modulation': 16,
        'NT': NT,
        'NR': NR,
        'snr_min': snr_k,
        'snr_max': snr_k+1.
    }
    params_list.append(params)

seq_len = 2
data_dir = './simulated_data'
map_filename = 'mappings.txt'

if __name__ == "__main__":
    file_oject = open(os.path.join(data_dir, map_filename), 'a')
    for params in params_list:
        produce_simulation(params, seq_len, data_dir, file_object)
    file_oject.close()    
