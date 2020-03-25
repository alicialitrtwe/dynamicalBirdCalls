from spectrogram import spectrogram
import numpy as np
from pwLogiRegression import pwLogiRegression
import os
from sklearn.decomposition import PCA
from scipy.signal import resample
import matplotlib.pyplot as plt
from soundsig.sound import plot_spectrogram
from soundsig.sound import plot_spectrogram
##changed freq_spacing from 50 to 100
def calculateSpectro(sound, spec_sample_rate=1000, freq_spacing = 100, min_freq=0, max_freq=10000):
    print(len(sound))
    t,f,spec,spec_rms = spectrogram(sound, 44100, spec_sample_rate=spec_sample_rate, freq_spacing=freq_spacing, \
		min_freq=min_freq, max_freq=max_freq, cmplx=True)
    spectro = 20*np.log10(np.abs(spec))
    return spectro


os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis')
allSegments = np.load('allSegment_Len500.npy', allow_pickle = True)
birdName = np.load('birdName.npy')
birdName = np.repeat(birdName, 3)
#allFund = np.load('fund_Len500.npy')
#nonanIndFund = ~np.isnan(allFund)
##lessInd = np.where(allFund<650)[0] 
#nonanInd = [d and m for d, m in zip(nonanIndFund, nonanIndFund)]
#allFundFiltered = allFund[nonanInd]
#birdName = birdName[nonanInd]
#allSegments = allSegments[nonanInd]

#tdata = np.arange(2000)
#num = [np.int(2000*allFundFiltered[i]/669) for i in range(len(allFundFiltered))]
#allResampledSeg = [resample(allSegments[i], num = num[i], t=tdata)[0] for i in range(len(allFundFiltered))]
#allResampledSeg = allSegments
#length = [len(allResampledSeg[i]) for i in range(len(allResampledSeg))]
segmentLength = 200 
#allResampledSeg = np.array([allResampledSeg[i][int(length[i]/2-segmentLength/2):int(length[i]/2+segmentLength/2)] for i in range(len(allFundFiltered))])
allSegments = np.array([allSegments[i][int(250-segmentLength/2):int(250+segmentLength/2)] for i in range(len(allSegments))])

#norm = 'Normed'
#if norm == 'Normed':
#	allResampledSeg = np.array([Segment/np.abs(Segment).max()*np.max(allResampledSeg) for Segment in allResampledSeg])
#allSpectrosRS = np.array([np.ravel(calculateSpectro(Seg)) for Seg in allResampledSeg])
#allSpectrosRS = allSpectrosRS - allSpectrosRS.max()


allSegments = np.array([Segment/np.abs(Segment).max() for Segment in allSegments])
allSpectros = np.array([np.ravel(calculateSpectro(Seg)) for Seg in allSegments])
allSpectros = allSpectros - allSpectros.max()


#(tDebug ,freqDebug ,specDebug , rms) = spectrogram(allSegments[9][950:1050], 44100, 1000.0, 200) 
##plot_spectrogram(tDebug, freqDebug, specDebug)
#plt.figure()
#plt.plot(allSegments[9][950:1050])
#
#(tDebug ,freqDebug ,specDebug , rms) = spectrogram(allResampledSeg[9], 44100, 1000.0, 200)
#plt.figure()
##plot_spectrogram(tDebug, freqDebug, specDebug)
#plt.plot(allResampledSeg[9])
#
#
norm = 'NormedTo1'
segmentLengthL = [200]
embDimL = [10]
tauL = [3]
PCA_n_components = 0 
pixels = 15
spread = 0.001
for Len in range(len(segmentLengthL)):
	for Dim in range(len(embDimL)):
		for Tau in range(len(tauL)):
			segmentLength = segmentLengthL[Len]
			embDim = embDimL[Dim]
			tau = tauL[Tau]
			if PCA_n_components < embDim and embDim*tau < 300:								
				allPIs = np.load('PI_Len%s_Dim%s_Tau%s_PCA%s_p%s_s%s_%s.npy' %(segmentLength, embDim, tau, PCA_n_components, pixels, spread, norm))

Ind = np.arange(len(birdName)) 
np.random.shuffle(Ind)
birdName = birdName[Ind]
#allSpectrosRS = allSpectrosRS[Ind]
allSpectros = allSpectros[Ind]
allPIs =allPIs[Ind]
##scores, pValues, nhighScores, ncounter = pwLogiRegression(allSpectros, birdName, nonanIndFeature=None, cv=True, printijScores=False)
##scores, pValues, nhighScores, ncounter = pwLogiRegression(allPIs, birdName, nonanIndFeature=None, cv=False, printijScores=False)
allFeaturesL = [allSpectros]#allPIs, 
ax = plt.figure().add_subplot(111)
ax.set_xlim(0, 2.5)
ax.set_ylim(20, 100)
for i in range(5):#(len(allFeaturesL)):
	scores, pValues, nhighScores, ncounter = pwLogiRegression(allFeaturesL[i], birdName, cv=True, printijScores=False)
	xRandom = np.random.random(len(scores))*0.6 + i
	ax.scatter(xRandom, scores, s = 10)
plt.show()
