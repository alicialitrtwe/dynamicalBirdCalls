import numpy as np
import os
from pwLogiRegression import pwLogiRegression

os.chdir('/Users/Alicia/Desktop/Freq_250_12000/BirdCallProject/dataAnalysis')


birdName = np.load('birdName.npy')
birdName = np.repeat(birdName, 1)
SegmentLength = 600
SegLen = 600
allSegments = np.load('allMidSegment_Len%d.npy' %SegLen)
allRMS = np.array([(np.mean(Segment**2))**(1/2) for Segment in allSegments])
allRMS = allRMS[:, None]
print(allRMS)
pwLogiRegression = pwLogiRegression(allRMS, birdName, nonanIndFeature=None, gridSearch=True, printijScores=True)