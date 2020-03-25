import os 
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from discriminate import discriminate

frequencyL, frequencyH = [20, 10000]
# Go to the folder that has the h5 files corresponding to the BioSound objects.
h5dir = '/Users/Alicia/Desktop/AdultVocalizations/h5files_%s_%s' %(frequencyL, frequencyH)
os.chdir(h5dir)

birdName = np.load('birdName_%s_%s.npy' %(frequencyL, frequencyH))

segmentLengthL = [400]
embDimL = [25]
tauL = [5]
PCA_n_componentsL = [10] 
pixelsL = [15]
spreadL = [2]
nDmaxL = [4]
savePD = True 
savePI = True
for sL in range(len(segmentLengthL)):
	for eD in range(len(embDimL)):
		for t in range(len(tauL)):
			for PCA in range(len(PCA_n_componentsL)):
				for p in range(len(pixelsL)):
					for s in range(len(spreadL)):
						for nD in range(len(nDmaxL)):
							segmentLength = segmentLengthL[sL]
							embDim = embDimL[eD]
							tau = tauL[t]
							PCA_n_components = PCA_n_componentsL[PCA]
							pixels = pixelsL[p]
							spread = spreadL[s]
							nDmax = nDmaxL[nD]

							#PD = np.load('PD_sL%s_eD%s_t%s_PCA%s.npy' %(segmentLength, embDim, tau, PCA_n_components))
							PI = np.load('PI_sL%s_eD%s_t%s_PCA%s_p%s_s%s.npy' %(segmentLength, embDim, tau, PCA_n_components, pixels, spread))
							print(PI.shape)
nDmax = 225
#birdName = np.hstack((birdName, birdName, birdName))
discriminate(PI[-607:], birdName, nDmax)