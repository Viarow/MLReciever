from sample_generator import produce_simulation
import numpy as np
import os

NT = 16
NR = 64
snr_min = 4.0
snr_max = 10.0
test_points = 200
intervals = np.linspace(snr_min, snr_max, test_points+1)

seq_len = 2
data_dir = './simulated_data'
map_filename = 'mappings.txt'

if __name__ == "__main__":
    file_object = open(os.path.join(data_dir, map_filename), 'a')
    params_list = []
    for k in np.arange(0, test_points, 1):
        params_k = {
            'modulation': 'QAM_16',
            'data': False,
            'NT': NT,
            'NR': NR,
            'snr_min': intervals[k],
            'snr_max': intervals[k+1]
        }
        params_list.append(params_k)

    for params in params_list:
        produce_simulation(params, seq_len, data_dir, file_object)
    file_object.close()    
