from dynamicalbirdcalls.find_spectrum import find_spectrum
from dynamicalbirdcalls.config import processed_data_dir
import os
import numpy as np
os.chdir(processed_data_dir)

segment_len = 300
segment_list = np.load('segment_lists/segment_list_%d.npy' %segment_len)
segment_list = segment_list.reshape(segment_list.shape[0]*segment_list.shape[1], segment_len)
spectrum_list = find_spectrum(segment_list, segment_len, Save=True) 
