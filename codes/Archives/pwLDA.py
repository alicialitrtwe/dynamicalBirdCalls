import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from discriminate import discriminate

def pwLDA(allFeatures, classes, nonanIndFeature=None, gridSearch=False, printijScores=False):
	# use only uniqueClassName with larger than 10 samples
	uniqueClassName, uniqueCNameCount = np.unique(classes, return_counts = True)  # uniqueClassName to be discriminated should be same as ldaMod.classes_
	goodClassNameInd = np.array([n >= 10 for n in uniqueCNameCount])
	goodSampleInd = np.array([b in uniqueClassName[goodClassNameInd] for b in classes])
	goodClassNameInd[np.where(uniqueClassName=='Unknown00F')[0]] = False

	if nonanIndFeature==None:
		goodInd = goodSampleInd
	else:
		goodInd = [c and f for c, f in zip(goodSampleInd, nonanIndFeature)]

	#take good indices and enode the classes labels as numbers
	goodClasses = classes[goodInd]
	goodFeatures = allFeatures[goodInd]
	le = preprocessing.LabelEncoder()
	le.fit(goodClasses)
	goodClassLabels = le.transform(goodClasses) 
	goodClassNameCount = uniqueClassName[goodClassNameInd].size
	scores = np.zeros((int(goodClassNameCount*(goodClassNameCount-1)/2), 3))


	ncounter = 0
	for i in range(goodClassNameCount):
		for j in range(goodClassNameCount):
			if i >= j:
				pass
			else:			
				featureij = np.vstack((goodFeatures[np.where(goodClassLabels == i)[0]],  goodFeatures[np.where(goodClassLabels == j)[0]] ))
				classij = np.hstack((goodClassLabels[np.where(goodClassLabels == i)[0]],  goodClassLabels[np.where(goodClassLabels == j)[0]] ))
				scores[ncounter] = discriminate(featureij, classij, nDmax=225)

				ncounter+=1
	print('mean scores: ', np.mean(scores, axis=0), 'ntotal: %s' %ncounter)	
	return scores, ncounter


