import numpy as np
import os
import matplotlib.pyplot as plt

from dynamicalbirdcalls.state_space_reconstruction import time_delay_reconstru
from dynamicalbirdcalls.config import processed_data_dir

os.chdir(processed_data_dir)
segment_list = np.load('segment_lists/segment_list_500.npy')

#allSegments = [segment[int(origLen/2-segmentLength/2) : int(origLen/2+segmentLength/2)] for segment in allSegments]
#allMIs = [MI_lcmin(segment) for segment in allSegments]
#print(np.mean(allMIs), allMIs) mean 3, mode 3

allReconstru = [TimeDelayReconstruct_allD(segment, 3, 20) for segment in allSegments]
E = np.mean([fnnCao(reconstru) for reconstru in allReconstru], axis=0)
print(E)
