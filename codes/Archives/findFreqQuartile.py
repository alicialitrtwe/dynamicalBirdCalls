from findSpectrum import findSpectrum
import os
import numpy as np
from pwLogiRegression import pwLogiRegression
maxFund = 1500
minFund = 300
lowFc = 200
highFc = 6000
minSaliency = 0.5
debugFig = 0
minFormantFreq = 500
maxFormantBW = 1000
method='Stack'
fs = 44100


os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis')
allSegments = np.load('allSegment_Len500.npy')
birdName = np.load('birdName.npy')
birdName = np.repeat(birdName, 3)

FreqQuartile = np.array([findSpectrum(Seg, fs, f_high=10000) for Seg in allSegments])
#FreqQuartile = FreqQuartile[:,:3]#/FreqQuartile[:,3][:, None]
print(FreqQuartile[:,:2])

Ind = np.arange(len(birdName)) 
np.random.shuffle(Ind)
birdName = birdName[Ind]
FreqQuartile = FreqQuartile[Ind]

scores, pValues, nhighScores, ncounter = pwLogiRegression(FreqQuartile, birdName, nonanIndFeature=None, cv=True, printijScores=False)
#discriminate(formant[nonanInd, :2], birdName[nonanInd], 2)