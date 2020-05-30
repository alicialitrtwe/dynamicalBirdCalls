import numpy as np
import matplotlib.pyplot as plt
from soundsig import detect_peaks
from scipy.signal import resample

def findPeriod(allSegments, segmentLength, norm, avgPeriodLen=100, SavePeriod=False):
#mean Periodlenth for SegLen 600: 65
	allExPeriod = []
	periodLen = []
	origLen = allSegments.shape[1]
	print('segment idx', int(origLen/2-segmentLength/2) , int(origLen/2+segmentLength/2))
	allSegments = allSegments[:, int(origLen/2-segmentLength/2) : int(origLen/2+segmentLength/2)]
	for isong in range(len(allSegments)):#
		soundSeg = allSegments[isong]/np.abs(allSegments[isong]).max() 
		if abs(min(soundSeg)) > max(soundSeg):
			ind = detect_peaks(soundSeg, mph=0.25, mpd=44.1, show=False, edge='both', kpsh=1, valley=1)
		else:
			ind = detect_peaks(soundSeg, mph=0.25, mpd=44.1, show=False, edge='both', kpsh=1, valley=0)
		periodStack = np.zeros((len(ind)-1, avgPeriodLen))
		for i in range(len(ind)-1):
			periodLen.append(ind[i+1] - ind[i])
			periodSeg = soundSeg[ind[i]:ind[i+1]]
			tdata = np.arange(len(periodSeg))
			(resampledPeriod, resampledtdata) = resample(periodSeg, num = avgPeriodLen, t = tdata)
			periodStack[i] = resampledPeriod
		avgPeriod = np.mean(periodStack, axis =0)
		diff = np.sum((periodStack - avgPeriod)**2, axis=1)
		exInd = np.where(diff == np.amax(diff))
		exPeriod = periodStack[exInd][0]
		if norm=='Normed':
			exPeriod = exPeriod/np.abs(exPeriod).max() 
		if norm =='unNormed':
			exPeriod = exPeriod/1
		allExPeriod.append(exPeriod)
	print('calculated avg period length: ', np.mean(periodLen))
	if SavePeriod==True:
		np.save('allPeriod_Len%d_%s.npy' %(SegmentLength, norm), allExPeriod)	
	return np.array(allExPeriod)
#import os
#os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis')
#allSegments = np.load('allSegment_Len500.npy')
#norm = 'Normed'
#allPeriod = findPeriod(allSegments, 500, norm)
#np.save('allPeriod_Len100_From500.npy', allPeriod)
#plt.figure()
#for i in range(len(allPeriod)):
#	plt.plot(allPeriod[i])
#plt.show()
#			

