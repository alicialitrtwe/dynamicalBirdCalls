import numpy as np
import os
from pwLogiRegression import pwLogiRegression
import matplotlib.pyplot as plt
from pwLDA import pwLDA
os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis')
#os.chdir('/Users/Alicia/Desktop/AdultVocalizations/h5files_20_10000')

birdName = np.load('birdName.npy')
birdName = np.repeat(birdName, 3)
#segmentLengthL = [300, 350, 400]
#embDimL = [10, 20, 30]
#tauL = [3, 6, 9, 12]
#PCA_n_componentsL = [0, 10] 
#pixelsL = [15, 20, 25, 30]
#spreadL = [1, 2]

spreadL = [1]
norm = 'Normed2Max'
segmentLengthL = [500]
embDimL = [25]
tauL = [3]
PCA_n_componentsL = [0] 
pixelsL = [15]
spreadL = [1]
discriminateScores = np.zeros((len(segmentLengthL), len(embDimL), len(tauL), len(PCA_n_componentsL), len(pixelsL), len(spreadL)))
for Len in range(len(segmentLengthL)):
	for Dim in range(len(embDimL)):
		for Tau in range(len(tauL)):
			for PCA in range(len(PCA_n_componentsL)):		
					for p in range(len(pixelsL)):
						for s in range(len(spreadL)):
							
							segmentLength = segmentLengthL[Len]
							embDim = embDimL[Dim]
							tau = tauL[Tau]
							PCA_n_components = PCA_n_componentsL[PCA]
							pixels = pixelsL[p]
							spread = spreadL[s]

							if PCA_n_components < embDim and embDim*tau < 300:
								print('\nprocessing: segmentLength', segmentLength, 'embDim ', embDim, ' tau ', tau, ' PCA_n_components ', PCA_n_components, ' pixels ', pixels, ' spread ', spread) 
								
								allPIs = np.load('PI_Len%s_Dim%s_Tau%s_PCA%s_p%s_s%s_%s.npy' %(segmentLength, embDim, tau, PCA_n_components, pixels, spread, norm))
								#discriminateScores[Len, Dim, Tau, PCA, p, s] = \
								#scores, score1, ncounter = pwLogiRegression(allPIs, birdName, nonanIndFeature=None, gridSearch=True, printijScores=False)
								scores, pValues, nhighScores, ncounter = pwLogiRegression(allPIs, birdName, nonanIndFeature=None, cv=True, printijScores=False)

#np.save('discriminateScores_Len400.npy',discriminateScores)
