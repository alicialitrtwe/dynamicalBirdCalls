import numpy as np
import os
from pwLogiRegression import pwLogiRegression
from findPeriods import findPeriod
from findSpectrum import findSpectrum
import matplotlib.pyplot as plt
from ripser import Rips
from persim import PersImage, plot_diagrams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.stats import binom
from AttractorReconstructUtilities import TimeSeries3D
from soundsig.sound import plot_spectrogram, spectrogram
os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis')

norm = 'Normed2Max'
segmentLengthL = [500]
embDimL = [10]
tauL = [3]
PCA_n_components = 0 
pixels = 15
spread = 1

for Len in range(len(segmentLengthL)):
	for Dim in range(len(embDimL)):
		for Tau in range(len(tauL)):
			segmentLength = segmentLengthL[Len]
			embDim = embDimL[Dim]
			tau = tauL[Tau]
			if PCA_n_components < embDim and embDim*tau < 300:								
				allPIs = np.load('PI_Len%s_Dim%s_Tau%s_PCA%s_p%s_s%s_%s_Final.npy' %(segmentLength, embDim, tau, PCA_n_components, pixels, spread, norm))
				allPDs = np.load('PD_Len%s_Dim%s_Tau%s_PCA%s_%s_Final.npy' %(segmentLength, embDim, tau, PCA_n_components, norm), allow_pickle=True)
#pim = PersImage(pixels=[15,15], spread=1)
#imgs = pim.transform(PD[0])
#fig = plt.figure()
#ax = fig.subplots()
#ax.imshow(imgs, cmap=plt.get_cmap("viridis"))
#plt.show()

birdName = np.load('birdName.npy')
birdName = np.repeat(birdName, 3)
allSegments = np.load('allSegment_Len500.npy')
allSpectrums = np.array([findSpectrum(Seg, 44100, f_high=22040) for Seg in allSegments])
allFunds = np.load('fund_Len500.npy')[:,None]
allFormants = np.load('formant_Len500.npy')[:, :2]
Ind = np.arange(len(birdName))
np.random.seed(41) 
np.random.shuffle(Ind)

birdName = birdName[Ind]
allFunds = allFunds[Ind]
allPIs = allPIs[Ind]
allPDs = allPDs[Ind]
allFormants = allFormants[Ind]
allSpectrums = allSpectrums[Ind]
allSegments = allSegments[Ind]

notNanFund = (np.sum(np.isnan(allFunds), axis = 1) == 0)
notNaNForm = (np.sum(np.isnan(allFormants), axis = 1) == 0)
notNaNInd = [d and m for d, m in zip(notNanFund, notNaNForm)]

classes = birdName
allFeatures = allPIs
nonanIndFeature = notNaNInd

cv=True
MINCOUNT = 10
cvFolds = 2
MINCOUNTTRAINING = 5

if nonanIndFeature is not None:
	classes = classes[nonanIndFeature]
	allFeatures = allFeatures[nonanIndFeature]
	allSegments = allSegments[nonanIndFeature]

uniqueClassName, uniqueCNameCount = np.unique(classes, return_counts = True)  # uniqueClassName to be discriminated should be same as lr.classes_
goodClassNameInd = np.array([n >= MINCOUNT for n in uniqueCNameCount])
goodSampleInd = np.array([b in uniqueClassName[goodClassNameInd] for b in classes])
goodClassNameInd[np.where(uniqueClassName=='Unknown00F')[0]] = False

#take good indices and enode the classes labels as numbers
goodClasses = classes[goodSampleInd]
goodFeatures = allFeatures[goodSampleInd]
goodSegments = allSegments[goodSampleInd]

le = preprocessing.LabelEncoder()
le.fit(goodClasses)
goodClassLabels = le.transform(goodClasses) 
nGoodClass = uniqueClassName[goodClassNameInd].size

lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
verbose=0, warm_start=False)


i=0
j=14

segmentij = np.vstack((goodSegments[np.where(goodClassLabels == i)[0]],  goodSegments[np.where(goodClassLabels == j)[0]] ))
featureij = np.vstack((goodFeatures[np.where(goodClassLabels == i)[0]],  goodFeatures[np.where(goodClassLabels == j)[0]] ))

PDs = np.concatenate((goodFeatures[np.where(goodClassLabels == i)[0]], goodFeatures[np.where(goodClassLabels == j)[0]]))

classij = np.hstack((goodClassLabels[np.where(goodClassLabels == i)[0]],  goodClassLabels[np.where(goodClassLabels == j)[0]] ))			

for n in range(len(classij)):
	print(classij[n])
	#inverse_image = np.copy(featureij[n]).reshape((15,15))
	#pim = PersImage(pixels=[15,15], spread=1)
	#pim.show(inverse_image)
	#plt.show()

	TimeSeries3D(segmentij[n], 3, 10, 'Name', axis_limit = None, elev = 90, azim = 0)
	#fig = plt.figure()
	#ax = fig.subplots()
	#plot_diagrams(PDs[n], ax=ax, xy_range=[0, 30000, 0, 30000], legend=False, lifetime=True)
	plt.show()
	#plt.figure()
	#(tDebug ,freqDebug ,specDebug , rms) = spectrogram(segmentij[n], 44100, 1000.0, 100, min_freq=0, max_freq=10000, nstd=6, cmplx=True) 
	#plot_spectrogram(tDebug, freqDebug, specDebug)
	#plt.show()
"""
if cv==True:
	cvCount = 0
	skf = StratifiedKFold(n_splits = cvFolds)
	skfList = skf.split(featureij, classij)
	for train, test in skfList:	
		# Enforce the MINCOUNT in each class for Training
		trainClasses, trainCount = np.unique(classij[train], return_counts=True)
		goodIndClasses = np.array([n >= MINCOUNTTRAINING for n in trainCount])
		goodIndTrain = np.array([b in trainClasses[goodIndClasses] for b in classij[train]])
		# Specity the training data set, the number of groups and priors
		yTrain = classij[train[goodIndTrain]]
		XrTrain = featureij[train[goodIndTrain]]
		trainClasses, trainCount = np.unique(yTrain, return_counts=True) 
		ntrainClasses = trainClasses.size
		# Skip this cross-validation fold because of insufficient data
		if ntrainClasses < 2:
			continue
		goodTrainInd = np.array([b in trainClasses for b in classij[test]])	

		if (goodTrainInd.size == 0):
			continue	
		lr.fit(XrTrain, yTrain)

		inverse_image = np.copy(lr.coef_).reshape((15,15))
		pim = PersImage(pixels=[15,15], spread=1)
		pim.show(inverse_image)
		plt.show()
"""