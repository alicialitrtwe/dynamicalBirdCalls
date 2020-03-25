from findSpectro import findSpectro
import os
import numpy as np
from sklearn.decomposition import PCA
from discriminate import discriminate

frequencyL, frequencyH = [20, 10000]
# Go to the folder that has the h5 files corresponding to the BioSound objects.
h5dir = '/Users/Alicia/Desktop/AdultVocalizations/h5files_%s_%s' %(frequencyL, frequencyH)
os.chdir(h5dir)

segmentLength = 400
nPCs = 10

birdName = np.load('birdName_%s_%s.npy' %(frequencyL, frequencyH))
allFirstSeg = np.load('allFirstSeg_%s_%s.npy' %(frequencyL, frequencyH))
allMidSeg = np.load('allMidSeg_%s_%s.npy' %(frequencyL, frequencyH))
allThirdSeg = np.load('allThirdSeg_%s_%s.npy' %(frequencyL, frequencyH))
allSeg = np.vstack((allFirstSeg, allMidSeg, allThirdSeg))
allSeg = allSeg[:, int(allSeg.shape[1]/2-segmentLength/2) : int(allSeg.shape[1]/2+segmentLength/2)]
f = np.array([np.ravel(findSpectro(Seg)[0]) for Seg in allSeg[:3]])
print(f)
#pca1 = PCA(n_components=nPCs)
#spectro_PCA = pca1.fit_transform(spectro) 
#np.save('spectro_allSeg.npy', spectro)
'''

segmentLengthL = [400]
embDimL = [25]
tauL = [5]
PCA_n_componentsL = [0] 
pixelsL = [15]
spreadL = [2]
for sL in range(len(segmentLengthL)):
	for eD in range(len(embDimL)):
		for t in range(len(tauL)):
			for PCA_nc in range(len(PCA_n_componentsL)):
				for p in range(len(pixelsL)):
					for s in range(len(spreadL)):
						segmentLength = segmentLengthL[sL]
						embDim = embDimL[eD]
						tau = tauL[t]
						PCA_n_components = PCA_n_componentsL[PCA_nc]
						pixels = pixelsL[p]
						spread = spreadL[s]

							#PD = np.load('PD_sL%s_eD%s_t%s_PCA%s.npy' %(segmentLength, embDim, tau, PCA_n_components))
						PI = np.load('PI_sL%s_eD%s_t%s_PCA%s_p%s_s%s.npy' %(segmentLength, embDim, tau, PCA_n_components, pixels, spread))

#pca2 = PCA(n_components=20)
#PI_PCA = pca2.fit_transform(PI[607:1214]) 
features = np.hstack((spectro_PCA, PI[607:1214]))
discriminate(spectro_PCA, birdName, 300)
discriminate(features, birdName, 300) '''