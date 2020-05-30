import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binom
import matplotlib.pyplot as plt
from persim import PersImage, plot_diagrams

def inverse(allFeatures, classes, nonanIndFeature=None, cv=False, printijScores=False):
	# use only uniqueClassName with larger than 10 samples
	MINCOUNT = 10
	cvFolds = 10
	MINCOUNTTRAINING = 5

	if nonanIndFeature is not None:
		classes = classes[nonanIndFeature]
		allFeatures = allFeatures[nonanIndFeature]

	uniqueClassName, uniqueCNameCount = np.unique(classes, return_counts = True)  # uniqueClassName to be discriminated should be same as lr.classes_
	goodClassNameInd = np.array([n >= MINCOUNT for n in uniqueCNameCount])
	goodSampleInd = np.array([b in uniqueClassName[goodClassNameInd] for b in classes])
	goodClassNameInd[np.where(uniqueClassName=='Unknown00F')[0]] = False

	#take good indices and enode the classes labels as numbers
	goodClasses = classes[goodSampleInd]
	goodFeatures = allFeatures[goodSampleInd]
	
	le = preprocessing.LabelEncoder()
	le.fit(goodClasses)
	goodClassLabels = le.transform(goodClasses) 
	nGoodClass = uniqueClassName[goodClassNameInd].size
	nPair = int(nGoodClass*(nGoodClass-1)/2)
	scores = np.zeros(nPair)
	pValues = np.zeros(nPair)

	ncounter = 0
	nhighScores = 0
	nSigP= 0

	lr = LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
	intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
	penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
    verbose=0, warm_start=False)
	#lt = [108, 110, 164,163,134,244,100,97,123,13,101,23,208,37]

	for i in range(nGoodClass):
		for j in range(nGoodClass):
			#if i < j:
			if (i ==0) & (j ==1):
				featureij = np.vstack((goodFeatures[np.where(goodClassLabels == i)[0]],  goodFeatures[np.where(goodClassLabels == j)[0]] ))
				classij = np.hstack((goodClassLabels[np.where(goodClassLabels == i)[0]],  goodClassLabels[np.where(goodClassLabels == j)[0]] ))			
				if cv==True:
					cvCount = 0
					lrYes = 0
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
						print(lr.coef_)

						#inverse_image = np.copy(lr.coef_).reshape((10,10))
						##np.save('inverseimage_0_14_p20_s0.01.npy', inverse_image)
						#pim = PersImage(pixels=[10,10], spread=1)
						#pim.show(inverse_image)
						#plt.show()
						
						lrYes += np.around((lr.score(featureij[test[goodTrainInd]], classij[test[goodTrainInd]]))*goodTrainInd.size)
						cvCount += goodTrainInd.size
					lrYesInt = int(lrYes)
					p = 1.0/2
					lrP = 0
					for k in range(lrYesInt, cvCount+1):
						lrP += binom.pmf(k, cvCount, p)

					print ("LR: %.2f %% (%d/%d p=%.4f)" % (100.0*lrYes/cvCount, lrYes, cvCount, lrP))					
					#if ncounter in lt:
					#	print(ncounter, i, j)
					scores[ncounter] = 100.0*lrYes/cvCount
					pValues[ncounter] = lrP
					if scores[ncounter] >=90.0 and pValues[ncounter] <=0.05:
						nhighScores+=1
					if pValues[ncounter] <=0.05:
						nSigP+=1


				else:
					X_train, X_test, y_train, y_test = train_test_split(featureij, classij, test_size=0.10, random_state=42)
					lr.fit(X_train, y_train)

					print(lr.coef_)

					#inverse_image = np.copy(lr.coef_).reshape((10,10))
					##np.save('inverseimage_0_14_p20_s0.01.npy', inverse_image)
					#pim = PersImage(pixels=[10,10], spread=1)
					#pim.show(inverse_image)
					#plt.show()

					scores[ncounter] = lr.score(X_test, y_test)
					if printijScores:
						print('classes:', i, j, 'scores', scores[ncounter])

					if scores[ncounter] >= .95:
						nhighScores+=1


				ncounter+=1
	print('mean scores: ', np.mean(scores), 'nhighScores/ntotal: %s/%s' %(nhighScores, ncounter), 'nSigP/ntotal: %s/%s' %(nSigP, nPair))	
	return scores, pValues, nhighScores, nPair

