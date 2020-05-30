import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from AttractorReconstructUtilities import TimeDelayReconstruct, TimeDelayReconstruct_allD, MI_lcmin, fnnCao
os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis')

allSegments = np.load('allSegment_Len500.npy')
origLen = 500
segmentLength = 500
#allSegments = [segment[int(origLen/2-segmentLength/2) : int(origLen/2+segmentLength/2)] for segment in allSegments]
#allMIs = [MI_lcmin(segment) for segment in allSegments]
#print(np.mean(allMIs), allMIs) mean 3, mode 3

allReconstru = [TimeDelayReconstruct_allD(segment, 3, 20) for segment in allSegments]
E = np.mean([fnnCao(reconstru) for reconstru in allReconstru], axis=0)
print(E)
