import numpy as np
import os
import matplotlib.pyplot as plt

from dynamicalbirdcalls.generate_pers_image import generate_PD, generate_PI
from dynamicalbirdcalls.state_space_reconstruction import time_delay_reconstru
from dynamicalbirdcalls.config import processed_data_dir

os.chdir(processed_data_dir)

period_list = np.load('TrainingData/period_list_500.npy')

pixels_list = [15]
spread_list = [0.01]
embed_dim_list = [10]
tau_list = [3]
savePD = True 
savePI = True

segment_list = period_list
segment_len = 66

for embed_dim in embed_dim_list:
	for tau in tau_list:

		print('Processing\n  segment length: %d \n  embedding dimension: %d \n  time delay: %d \n' %(segment_len, embed_dim, tau))
		reconstru_list = [time_delay_reconstru(segment, embed_dim, tau) for segment in segment_list]
		PD_list = generate_PD(reconstru_list, embed_dim, tau)
		if savePD == True:
			np.save('TrainingData/PeriodPD_Len%s_Dim%s_Tau%s.npy' %(segment_len, embed_dim, tau), PD_list)

		for pixels in pixels_list:
			for spread in spread_list:

				print('  pixels: %d\n  spread: %s\n' %(pixels, spread)) 
				#PD_list = np.load('TrainingData/PD_Len%s_Dim%s_Tau%s.npy' %(segment_len, embed_dim, tau), allow_pickle=True)
				PI_list = generate_PI(PD_list, embed_dim, tau, pixels, spread)
				if savePI == True:
					np.save('TrainingData/PeriodPI_Len%s_Dim%s_Tau%s_Pix%s_Spd_%s.npy' %(segment_len, embed_dim, tau, pixels, spread), PI_list)
 

