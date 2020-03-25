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
def calculateSpectro(sound, spec_sample_rate=1000, freq_spacing = 210, min_freq=0, max_freq=10000):
    t,f,spec,spec_rms = spectrogram(sound, 44100, spec_sample_rate=spec_sample_rate, freq_spacing=freq_spacing, \
		min_freq=min_freq, max_freq=max_freq, cmplx=True)
    spectro = 20*np.log10(np.abs(spec))
    return spectro

os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis')
allSegments = np.load('allSegment_Len500.npy')
allPeriods = np.load('allPeriod_Len100_From500.npy')
birdName = np.load('birdName.npy')
birdName = np.repeat(birdName, 3)
allSegments = allPeriods
norm = 'Normed'
if norm == 'Normed':
	allSegments = np.array([Segment/np.abs(Segment).max()*np.max(allSegments) for Segment in allSegments])
allSpectros = np.array([np.ravel(calculateSpectro(Seg)) for Seg in allSegments])
allSpectros = allSpectros - allSpectros.max()

allFund = np.load('fund_Len500.npy')
nonanIndFund = ~np.isnan(allFund)
#lessInd = np.where(allFund<800)[0] 
nonanInd = [d and m for d, m in zip(nonanIndFund, nonanIndFund)]
#allFundFiltered = allFund[nonanInd]
##print(allFundFiltered[:100])
birdName = birdName[nonanInd]
allSpectros = allSpectros[nonanInd]
(tDebug ,freqDebug ,specDebug , rms) = spectrogram(allPeriods[0], 44100, 1000.0, 210, min_freq=0, max_freq=10000, nstd=2, cmplx=True) 
plot_spectrogram(tDebug, freqDebug, specDebug)
plt.show()
Ind = np.arange(len(birdName)) 
np.random.shuffle(Ind)
birdNameShuffled = birdName[Ind]
allSpectros =allSpectros[Ind]
#scores, pValues, nhighScores, ncounter = pwLogiRegression(allSpectros, birdNameShuffled, nonanIndFeature=None, cv=True, printijScores=False)
