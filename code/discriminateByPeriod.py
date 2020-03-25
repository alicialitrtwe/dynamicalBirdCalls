import numpy as np
import os
from pwLogiRegression import pwLogiRegression
from findPeriods import findPeriod
import matplotlib.pyplot as plt

os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis')
#os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_20_10000/dataAnalysis')
allSegments = np.load('allSegment_Len500.npy')*5000
allPeriods = np.load('allPeriod_Len80_From500.npy')
birdName = np.load('birdName.npy')
birdName = np.repeat(birdName, 3)
allFund = np.load('fund_Len500.npy')
nonanIndFund = ~np.isnan(allFund)
#lessInd = np.where(allFund<800)[0] 
nonanInd = [d and m for d, m in zip(nonanIndFund, nonanIndFund)]
#allFundFiltered = allFund[nonanInd]
##print(allFundFiltered[:100])
birdName = birdName[nonanInd]
allPeriods = allPeriods[nonanInd]

SegmentLengthL= [500]
norm = 'Normed'


for i in range(len(SegmentLengthL)):
	#allPeriods = findPeriods(allSegments, SegmentLengthL[i], norm)
	scores, pValues, nhighScores, ncounter = pwLogiRegression(allPeriods, birdName, nonanIndFeature=None, cv=True, printijScores=True)




