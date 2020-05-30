from AttractorReconstructUtilities import TimeDelayReconstruct
from sklearn.decomposition import PCA
import numpy as np

def timeDelayPCA(allSegment, segmentLength, embDim, tau, PCA_n_components=0):
	origLen = allSegment.shape[1]
	print('segment idx', int(origLen/2-segmentLength/2) , int(origLen/2+segmentLength/2))
	allSegment = allSegment[:, int(origLen/2-segmentLength/2) : int(origLen/2+segmentLength/2)]
	alltimeDelayedSeg = [TimeDelayReconstruct(Segment, Dimension=embDim, tauIndex=tau) for Segment in allSegment]
	if PCA_n_components != 0: 
		pca = PCA(n_components=PCA_n_components)
		alltimeDelayPCA = [pca.fit_transform(timeDelayedSeg) for timeDelayedSeg in alltimeDelayedSeg]
	else:
   		alltimeDelayPCA = alltimeDelayedSeg
	return np.array(alltimeDelayPCA)
