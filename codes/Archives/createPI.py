import numpy as np
import os
from sklearn.decomposition import PCA

os.chdir('/Users/Alicia/AnacondaProjects/CurrentProject')
from generatePersImage import generatePD, generatePI
from timeDelayPCA import timeDelayPCA


os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis')

allSegments = np.load('allSegment_Len500.npy')[:2]
norm = 'Normed2Max'
allSegments = np.array([Segment/np.abs(Segment).max()*np.max(allSegments) for Segment in allSegments])[:2]
#allSegments = np.array([Segment/np.abs(Segment).max() for Segment in allSegments])
#segmentLengthL = [500]
##segmentLengthL = [300, 350, 400]
##segmentLengthL = [400, 450]
##segmentLengthL = [450, 500, 550, 600]
##embDimL = [10, 20, 30]
#embDimL = [25]
##tauL = [3, 6, 9, 12]
#tauL = [2]
##PCA_n_componentsL = [0, 10] 
PCA_n_componentsL = [0] 
pixelsL = [15]
##pixelsL = [15,20,25,30]
##spreadL = [1, 2]
spreadL = [1]
norm = 'Normed2Max'
segmentLengthL = [500]
embDimL = [25]
tauL = [3]
PCA_n_components = 0 
pixels = 15
spread = 1
savePD = False 
savePI = False

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
						allPDs = generatePD(allTimeDelayedSeg, segmentLength, embDim, tau, PCA_n_components, savePD, norm)
			
						for p in range(len(pixelsL)):
							for s in range(len(spreadL)):
								pixels = pixelsL[p]
								spread = spreadL[s]
								#if 'MidPD_Len%s_Dim%s_Tau%s_PCA%s.npy' %(segmentLength, embDim, tau, PCA_n_components)
								print('processing segmentLength', segmentLength, 'embDim ', embDim, ' tau ', tau, ' PCA_n_components ', PCA_n_components, ' pixels ', pixels, ' spread ', spread, '\n\n') 
								PI0 = generatePI(allPDs, segmentLength, embDim, tau, PCA_n_components, pixels, spread, savePI, norm)
print(PI0[0])
allPIs = np.load('PI_Len%s_Dim%s_Tau%s_PCA%s_p%s_s%s_%s.npy' %(segmentLength, embDim, tau, PCA_n_components, pixels, spread, norm))
print(allPIs[0])