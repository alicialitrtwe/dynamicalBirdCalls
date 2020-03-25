from sklearn.svm import SVC
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binom

def pwSVM(allFeatures, classes, nonanIndFeature=None, cv=False, printijScores=False):
	# use only uniqueClassName with larger than 10 samples
	MINCOUNT = 10
	cvFolds = 10
	MINCOUNTTRAINING = 5

	uniqueClassName, uniqueCNameCount = np.unique(classes, return_counts = True)  # uniqueClassName to be discriminated should be same as lr.classes_
	goodClassNameInd = np.array([n >= MINCOUNT for n in uniqueCNameCount])
	goodSampleInd = np.array([b in uniqueClassName[goodClassNameInd] for b in classes])
	goodClassNameInd[np.where(uniqueClassName=='Unknown00F')[0]] = False

	if nonanIndFeature is None:
		goodInd = goodSampleInd
	else:
		goodInd = [c and f for c, f in zip(goodSampleInd, nonanIndFeature)]

	#take good indices and enode the classes labels as numbers
	goodClasses = classes[goodInd]
	goodFeatures = allFeatures[goodInd]

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

	clf = SVC(kernel='rbf', C = 1)

	for i in range(nGoodClass):
		for j in range(nGoodClass):
			if i < j:

				featureij = np.vstack((goodFeatures[np.where(goodClassLabels == i)[0]],  goodFeatures[np.where(goodClassLabels == j)[0]] ))
				classij = np.hstack((goodClassLabels[np.where(goodClassLabels == i)[0]],  goodClassLabels[np.where(goodClassLabels == j)[0]] ))			

				cvCount = 0
				svmYes = 0
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
					clf.fit(XrTrain, yTrain)
					svmYes += (clf.score(featureij[test[goodTrainInd]], classij[test[goodTrainInd]]))*goodTrainInd.size
					cvCount += goodTrainInd.size
				svmYesInt = int(np.around(svmYes))
				p = 1.0/2
				svmP = 0
				for k in range(svmYesInt, cvCount+1):
					svmP += binom.pmf(k, cvCount, p)

				print ("LR: %.2f %% (%d/%d p=%.4f)" % (100.0*svmYes/cvCount, svmYes, cvCount, svmP))					
				scores[ncounter] = 100.0*svmYes/cvCount
				pValues[ncounter] = svmP
				if scores[ncounter] >=95.0:
					nhighScores+=1
				if pValues[ncounter] <=0.05:
					nSigP+=1

				ncounter+=1
	print('mean scores: ', np.mean(scores), 'nhighScores/ntotal: %s/%s' %(nhighScores, ncounter), 'nSigP/ntotal: %s/%s' %(nSigP, nPair))	
	return scores, pValues, nhighScores, ncounter