import os 
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

SegmentLength = 400
allExPeriod = np.load('/Users/Alicia/Desktop/BirdCallProject/dataAnalysis/allExPeriod_Len%d.npy' %SegmentLength)

birdName = np.load('birdName.npy')
birdName = np.hstack((birdName, birdName, birdName))

allFirstSeg = np.load('allFirstSeg_%s_%s.npy' %(frequencyL, frequencyH))
allMidSeg = np.load('allMidSeg_%s_%s.npy' %(frequencyL, frequencyH))
allThirdSeg = np.load('allThirdSeg_%s_%s.npy' %(frequencyL, frequencyH))
allSeg = np.vstack((allFirstSeg, allMidSeg, allThirdSeg))

#spectro = spectro[607:]
#origLen = allMidSeg.shape[1]
#allMidSeg = allMidSeg[:, int(origLen/2-segmentLength/2) : int(origLen/2+segmentLength/2)]
#pca = PCA(n_components=20)
#allMidSeg = pca.fit_transform(allMidSeg)

segmentLength = 400
embDim = 25
tau = 5
PCA_n_components = 0
pixels = 15
spread = 2
nDmax = 4
PI = np.load('PI_sL%s_eD%s_t%s_PCA%s_p%s_s%s.npy' %(segmentLength, embDim, tau, PCA_n_components, pixels, spread))
#PI = PI[607:]
spectro = np.load('spectro_allSeg.npy')
allFreqQuartile = np.load('allSpectEnve_1024.npy')
allFormant = np.load('allFormant_1024.npy')[:, :2]
allFund = np.load('allFund_1024.npy')[:, :1]

# use only classes with larger than 10 samples
classes, classesCount = np.unique(birdName, return_counts = True)  # Classes to be discriminated should be same as ldaMod.classes_
goodIndClasses = np.array([n >= 10 for n in classesCount])
goodClassInd = np.array([b in classes[goodIndClasses] for b in birdName])
nonanIndFor = (np.sum(np.isnan(allFormant[:, :2]), axis = 1) == 0)
nonanIndFund = (np.sum(np.isnan(allFund[:, :1]), axis = 1) == 0)
nonanIndFreqQ = (np.sum(np.isnan(allFreqQuartile), axis = 1) == 0)
goodInd = goodClassInd
#goodInd = [g and For and Fund for g, For, Fund in zip(goodClassInd, nonanIndFor, nonanIndFund)]
#goodInd = [g and Fund for g, Fund in zip(goodClassInd, nonanIndFreqQ)]

birdNameGood = birdName[goodInd]
PIGood = PI[goodInd]
spectroGood = spectro[goodInd]
allSegGood = allSeg[goodInd]
allFormantGood = allFormant[goodInd]
allFundGood = allFund[goodInd]
allFreqQGood = allFreqQuartile[goodInd]
#allFormantGood = allFormantGood/allFundGood[:, 0][:, None]
#allFreqQGood = np.hstack((allFreqQGood, allFundGood[:, 0][:, None]))

pca1 = PCA(n_components=20)
spectroGood = pca1.fit_transform(spectroGood) 

le = preprocessing.LabelEncoder()
le.fit(birdNameGood)
birdLabelGood = le.transform(birdNameGood) 

classes, classesCount = np.unique(birdLabelGood, return_counts = True) 
nClasses = classes.size 
PIscores = np.zeros(int(nClasses*(nClasses-1)/2))
bestPIscores = np.zeros(int(nClasses*(nClasses-1)/2))
spectroscores = np.zeros(int(nClasses*(nClasses-1)/2))
formantscores = np.zeros(int(nClasses*(nClasses-1)/2))
freqQscores = np.zeros(int(nClasses*(nClasses-1)/2))
fundscores = np.zeros(int(nClasses*(nClasses-1)/2))
ncounter = -1

for i in range(nClasses):
	for j in range(nClasses):
		if i >= j:
			pass
		else:
			ncounter+=1
			twoClassPI = np.vstack((PIGood[np.where(birdLabelGood == i)[0]],  PIGood[np.where(birdLabelGood == j)[0]] ))
			twoClassSpectro = np.vstack((spectroGood[np.where(birdLabelGood == i)[0]],  spectroGood[np.where(birdLabelGood == j)[0]] ))
			twoClassSeg = np.vstack((allSegGood[np.where(birdLabelGood == i)[0]],  allSegGood[np.where(birdLabelGood == j)[0]] ))
			twoClassFormant = np.vstack((allFormantGood[np.where(birdLabelGood == i)[0]],  allFormantGood[np.where(birdLabelGood == j)[0]] ))
			twoClassFreqQ = np.vstack((allFreqQGood[np.where(birdLabelGood == i)[0]],  allFreqQGood[np.where(birdLabelGood == j)[0]] ))
			twoClassFund = np.vstack((allFundGood[np.where(birdLabelGood == i)[0]],  allFundGood[np.where(birdLabelGood == j)[0]] ))
			twoClassLabel = np.hstack((birdLabelGood[np.where(birdLabelGood == i)[0]],  birdLabelGood[np.where(birdLabelGood == j)[0]] ))
			#print('class:', i, j, 'number', len(np.where(birdLabelGood == i)[0]), len(np.where(birdLabelGood == j)[0]))
			
			twoClassFeatures = np.hstack((twoClassPI, twoClassFreqQ))
			X_train, X_test, y_train, y_test = train_test_split(twoClassFeatures, twoClassLabel, test_size=0.10, random_state=42)
			
			lrPI = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
			lrPI.fit(X_train[:, :225], y_train)
			PIscores[ncounter] = lrPI.score(X_test[:, :225], y_test)

			lrFreqQ = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
			lrFreqQ.fit(X_train[:, 225:], y_train)
			freqQscores[ncounter] = lrFreqQ.score(X_test[:, 225:], y_test)

			#lrPIGrid = LogisticRegression(class_weight=None, dual=False, fit_intercept=True, \
			#intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, \
			#random_state=0, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
			#param_grid = [{'penalty' : ['l1', 'l2'], 'C' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000] }]
			#clf = GridSearchCV(lrPIGrid, param_grid = param_grid, cv = 5, verbose=False, n_jobs=1)
			#best_clf = clf.fit(X_train[:, :225], y_train)
			#bestPIscores[ncounter] = best_clf.score(X_test[:, :225], y_test)
			"""
			lrSpectro = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
			lrSpectro.fit(X_train[:, 225:], y_train)
			spectroscores[ncounter] = lrSpectro.score(X_test[:, 225:], y_test)

			lrFund = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
			lrFund.fit(X_train[:, 225:], y_train)
			fundscores[ncounter] = lrFund.score(X_test[:, 225:], y_test)
			
			lrFormant = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
			lrFormant.fit(X_train[:, 225:], y_train)
			formantscores[ncounter] = lrFormant.score(X_test[:, 225:], y_test)

			
			lrSeg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
			lrSeg.fit(X_train[:, 225:], y_train)
			print('Seg scores:', lrSeg.score(X_test[:, 225:], y_test), '\n' )



"""
print(np.mean(PIscores), np.mean(freqQscores))
