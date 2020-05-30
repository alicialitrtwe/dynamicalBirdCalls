import numpy as np
import os
from sklearn.decomposition import PCA
from findPeriods import findPeriod
os.chdir('/Users/Alicia/AnacondaProjects/CurrentProject')
from generatePersImage import generatePD, generatePI
from timeDelayPCA import timeDelayPCA


os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis')

allSegments = np.load('allSegment_Len500.npy')
norm = 'Normed'
#SegmentLength = 500
#allSegments = findPeriod(allSegments, SegmentLength, norm, avgPeriodLen=65)
#norm = 'Period'
allSegments = np.array([Segment/np.abs(Segment).max() for Segment in allSegments])
PCA_n_componentsL = [0] 
pixelsL = [15]
spreadL = [0.001]#
#segmentLengthL = [65]
segmentLengthL = [300]
embDimL = [10]
tauL = [3]
savePD = True 
savePI = True

for Len in range(len(segmentLengthL)):
	for Dim in range(len(embDimL)):
		for Tau in range(len(tauL)):
			for PCA in range(len(PCA_n_componentsL)):

					segmentLength = segmentLengthL[Len]
					embDim = embDimL[Dim]
					tau = tauL[Tau]
					PCA_n_components = PCA_n_componentsL[PCA]
					if PCA_n_components < embDim and embDim*tau < 300:
						                                                                    
						allTimeDelayedSeg = timeDelayPCA(allSegments, segmentLength, embDim, tau, PCA_n_components)
						allPDs = generatePD(allTimeDelayedSeg, segmentLength, embDim, tau, PCA_n_components, norm, savePD)
						#allPDs = np.load('PD_Len%s_Dim%s_Tau%s_PCA%s_%s.npy' %(segmentLength, embDim, tau, PCA_n_components, norm), allow_pickle=True)
			
						for p in range(len(pixelsL)):
							for s in range(len(spreadL)):
								pixels = pixelsL[p]
								spread = spreadL[s]
								#if 'MidPD_Len%s_Dim%s_Tau%s_PCA%s.npy' %(segmentLength, embDim, tau, PCA_n_components)
								print('processing segmentLength', segmentLength, 'embDim ', embDim, ' tau ', tau, ' PCA_n_components ', PCA_n_components, ' pixels ', pixels, ' spread ', spread, '\n\n') 
								PI = generatePI(allPDs, segmentLength, embDim, tau, PCA_n_components, pixels, spread, norm, savePI)
#allPIs = np.load('PI_Len%s_Dim%s_Tau%s_PCA%s_p%s_s%s_%s.npy' %(segmentLength, embDim, tau, PCA_n_components, pixels, spread, norm))
#print(allPIs[0])