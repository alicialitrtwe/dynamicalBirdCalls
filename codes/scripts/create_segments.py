from soundsig.sound import temporal_envelope
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from dynamicalbirdcalls.find_time_quartile import find_time_quartile
from dynamicalbirdcalls.config import processed_data_dir

os.chdir(processed_data_dir)

if not os.path.exists('segment_lists'):
	os.makedirs('segment_lists')

fs = 44100
# try different short segment lengths
bird_name_list = [] 
segment_list_300 = [] 
segment_list_400 = [] 
segment_list_500 = [] 

# Create .npy files of lists of all short segments.
isound = 0    
for fname in sorted(os.listdir('./filtered_calls')):
	if fname.endswith('.npy'):

		isound += 1;     
		print('Processing filtered call %d: %s' %(isound, fname))   	     
		filtered_call = np.load('./filtered_calls/%s' %fname)  
		quartile_time = find_time_quartile(filtered_call, fs)
		bird_name = fname[0:10]
		bird_name_list.append(bird_name)
		segment_300 = []
		segment_400 = []
		segment_500 = []
		for i in range(len(quartile_time)):
		# take three segments from respectively the 25%, 50%, 75% of the amplitude envelope
			segment_300.append(filtered_call[quartile_time[i]-int(300/2) : quartile_time[i]+int(300/2)])
			segment_400.append(filtered_call[quartile_time[i]-int(400/2) : quartile_time[i]+int(400/2)])
			segment_500.append(filtered_call[quartile_time[i]-int(500/2) : quartile_time[i]+int(500/2)])
		
		segment_list_300.append(segment_300)
		segment_list_400.append(segment_400)
		segment_list_500.append(segment_500)

#save in .npy files
np.save('segment_lists/segment_list_300.npy', np.array(segment_list_300))
np.save('segment_lists/segment_list_400.npy', np.array(segment_list_400))
np.save('segment_lists/segment_list_500.npy', np.array(segment_list_500))
np.save('segment_lists/bird_name_list.npy', bird_name_list)