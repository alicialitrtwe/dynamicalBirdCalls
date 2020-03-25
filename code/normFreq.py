import numpy as np
import matplotlib.pyplot as plt
from detect_peaks import detect_peaks
from scipy.signal import resample
from scipy.signal import argrelextrema
import os
from soundsig.sound import WavFile, plot_spectrogram, spectrogram
os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis')
#os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_20_10000/dataAnalysis')
allSegments = np.load('allSegment_Len500.npy')
birdName = np.load('birdName.npy')
birdName = np.repeat(birdName, 3)
allFund = np.load('fund_Len500.npy')[:,None]
allFundFiltered = allFund[~np.isnan(allFund)] 
tdata = np.arange(500)
num = np.int(500/allFundFiltered[0]*669)
(resampledSeg, resampledtdata) = resample(allSegments[0], num = num, t = tdata)
#plt.plot(tdata, allSegments[0])
#plt.plot(resampledtdata, resampledSeg, '--')

(tDebug ,freqDebug ,specDebug , rms) = spectrogram(allSegments[0], 44100, 1000.0, 100, min_freq=0, max_freq=10000, nstd=6, cmplx=True) 
plot_spectrogram(tDebug, freqDebug, specDebug)


(tDebug ,freqDebug ,specDebug , rms) = spectrogram(resampledSeg, 44100, 1000.0, 100, min_freq=0, max_freq=10000, nstd=6, cmplx=True)
plt.figure()
plot_spectrogram(tDebug, freqDebug, specDebug)
#plt.plot(resampledtdata, resampledSeg)
#plt.figure()
#plt.plot(range(500), allSegments[0])
plt.show()