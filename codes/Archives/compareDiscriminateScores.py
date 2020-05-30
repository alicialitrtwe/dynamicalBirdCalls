import numpy as np
import os 

os.chdir('/Users/Alicia/Desktop/BirdCallProject/dataAnalysis')
discriminateScores = np.load('discriminateScores_Len400.npy')

segmentLengthL = [300, 350, 400]
embDimL = [10, 20, 30]
tauL = [3, 6, 9, 12]
PCA_n_componentsL = [0, 10] 
pixelsL = [15, 20, 25, 30]
spreadL = [1, 2]

for Len in range(len(segmentLengthL)):
	for Dim in range(len(embDimL)):
		for Tau in range(3):
			for PCA in range(1): #range(len(PCA_n_componentsL)):		
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
							print(np.where(discriminateScores[Len, Dim, Tau, PCA, :, 0]==np.max(discriminateScores[Len, Dim, Tau, PCA, :, 0])),discriminateScores[Len, 1, Tau, PCA, p, 0]>discriminateScores[Len, 0, Tau, PCA, p, 0],\
								discriminateScores[Len, 2, Tau, PCA, p, 0]>discriminateScores[Len, 0, Tau, PCA, p, 0], discriminateScores[Len, Dim, Tau, PCA, :, 0])
