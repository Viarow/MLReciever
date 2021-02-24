from sample_generator import produce_simulation
from numpy as np
import os

NT = 16
NR = 64
params_list = []
snr_min = 4.0
snr_max = 10.0
test_points = 200
intervals = numpy.linspace(snr_min, snr_max, test_points+1)

params_list = []
for k in range(0, testing_points):
    params_k = {
        'modulation': 'QAM_16',
        'NT': NT,
        'NR': NR,
        'snr_min': intervals[k],
        'snr_max': intervals[k+1]
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
