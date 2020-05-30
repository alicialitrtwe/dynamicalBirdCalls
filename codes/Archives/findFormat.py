from fundEstimator import fundEstimator
from formantEst import formantEst
import os
import numpy as np
from pwLogiRegression import pwLogiRegression
from pwSVM import pwSVM

maxFund = 1500
minFund = 300
lowFc = 200
highFc = 6000
minSaliency = 0.5
debugFig = 0
minFormantFreq = 300
maxFormantBW = 1000
method='Stack'
fs = 44100

os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis')
allSegments = np.load('allSegment_Len500.npy')

birdName = np.load('birdName.npy')
birdName = np.repeat(birdName, 3)

Ind = np.arange(len(birdName)) 
np.random.shuffle(Ind)
birdName = birdName[Ind]
#results = np.array([formantEst(Seg, fs, win1=True, debugFig = debugFig, maxFund = maxFund, minFund = minFund, lowFc = lowFc, highFc = highFc, minSaliency = minSaliency, minFormantFreq = minFormantFreq, maxFormantBW = maxFormantBW, method = method) for Seg in allSegments])
#np.save('fund_Len500_Win1.npy', results[:,0])
#np.save('formant_Len500_Win1.npy', results)
#allFund = np.load('fund_Len500.npy')
allFormant = np.load('formant_Len500_Win1.npy')[:,:3]
#allFormant = np.load('formant_Len500.npy')[:,:2]
#allFundRand = allFund[Ind]
#nonanIndFund = (np.isnan(allFund) == 0)
nonanIndForm = (np.sum(np.isnan(allFormant), axis = 1) == 0)
#pwLogiRegression(allFund[:,None], birdName, nonanIndFeature=nonanIndFund, cv=True, printijScores=False)
#pwLogiRegression(allFormant, birdName, nonanIndFeature=nonanIndForm, cv=True, printijScores=False)
pwSVM(allFormant, birdName, nonanIndFeature=nonanIndForm, cv=True, printijScores=False)