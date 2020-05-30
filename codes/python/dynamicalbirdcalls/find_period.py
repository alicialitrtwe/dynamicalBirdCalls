import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from dynamicalbirdcalls.detect_peaks import detect_peaks


def find_period(segment_list, segment_len, avg_period_len=66, Save=False):
# mean Periodlenth for SegLen 600: 65
	example_period_list = []
	period_len = []
	for isong in range(len(segment_list)):#
		segment = segment_list[isong]/np.abs(segment_list[isong]).max() 
		ind = detect_peaks(segment, mpd=44.1, show=False, edge='both', valley=1)
		period_stack = np.zeros((len(ind)-1, avg_period_len))
		for i in range(len(ind)-1):
			period_len.append(ind[i+1] - ind[i])
			period = segment[ind[i]:ind[i+1]]
			tdata = np.arange(len(period))
			(resam_period, _) = resample(period, num = avg_period_len, t = tdata)
			period_stack[i] = resam_period
		avg_period = np.mean(period_stack, axis =0)
		diff = np.sum((period_stack - avg_period)**2, axis=1)
		exInd = np.where(diff == np.amin(diff))
		example_period = period_stack[exInd][0]
		#plt.plot(example_period)
		#plt.plot(avg_period)
		#plt.show()
		example_period = example_period/np.abs(example_period).max() 
		example_period_list.append(example_period)

	print('calculated avg period length: ', np.mean(period_len))
	if Save==True:
		np.save('TrainingData/period_list_%d.npy' %(segment_len), example_period_list)	
	return np.array(example_period_list)

