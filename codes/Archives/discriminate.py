import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binom

def discriminate(X, y, nDmax):
    CVFOLDS = 10
    MINCOUNT = 10
    MINCOUNTTRAINING = 5

    # Initialize Variables and clean up data
    classes, classesCount = np.unique(y, return_counts = True)  # Classes to be discriminated should be same as ldaMod.classes_
    goodIndClasses = np.array([n >= MINCOUNT for n in classesCount])
    goodInd = np.array([b in classes[goodIndClasses] for b in y])

    yGood = y[goodInd]
    XGood = X[goodInd]
        
    classes, classesCount = np.unique(yGood, return_counts = True) 
    nClasses = classes.size         # Number of classes or groups 

    cvFolds = min(min(classesCount), CVFOLDS)
    if (cvFolds < CVFOLDS):
        print ('Warning in ldaPlot: Cross-validation performed with %d folds (instead of %d)' % (cvFolds, CVFOLDS))


    # Data size and color values   
    nD = XGood.shape[1]                 # number of features in X
    nX = XGood.shape[0]                 # number of data points in X

    # Use a uniform prior 
    myPrior = np.ones(nClasses)*(1.0/nClasses)  

    # Perform a PCA for dimensionality reduction so that the covariance matrix can be fitted.
    # nDmax = int(np.fix(np.sqrt(nX//5)))
    if nDmax < nD:
        print ('Warning: Insufficient data for', nD, 'parameters. PCA projection to', nDmax, 'dimensions.' )
    nDmax = min(nD, nDmax)
    pca = PCA(n_components=nDmax)
    Xr = pca.fit_transform(XGood)
    print ('Variance explained is %.2f%%' % (sum(pca.explained_variance_ratio_)*100.0))
    

    # Initialise Classifiers  
    ldaMod = LDA(n_components = min(nDmax,nClasses-1), priors = myPrior, shrinkage = None, solver = 'svd') 
    qdaMod = QDA(priors = myPrior)
    rfMod = RF()   # by default assumes equal weights
        
    # Perform CVFOLDS fold cross-validation to get performance of classifiers.
    ldaYes = 0
    qdaYes = 0
    rfYes = 0
    cvCount = 0

    skf = StratifiedKFold(n_splits = cvFolds)
    skfList = skf.split(Xr, yGood)
    
    for train, test in skfList:
        
        # Enforce the MINCOUNT in each class for Training
        trainClasses, trainCount = np.unique(yGood[train], return_counts=True)
        goodIndClasses = np.array([n >= MINCOUNTTRAINING for n in trainCount])
        goodIndTrain = np.array([b in trainClasses[goodIndClasses] for b in yGood[train]])

        # Specity the training data set, the number of groups and priors
        yTrain = yGood[train[goodIndTrain]]
        XrTrain = Xr[train[goodIndTrain]]

        trainClasses, trainCount = np.unique(yTrain, return_counts=True) 
        ntrainClasses = trainClasses.size
        
        # Skip this cross-validation fold because of insufficient data
        if ntrainClasses < 2:
            continue
        goodInd = np.array([b in trainClasses for b in yGood[test]])    
        if (goodInd.size == 0):
            continue
           
        # Fit the data
        trainPriors = np.ones(ntrainClasses)*(1.0/ntrainClasses)
        ldaMod.priors = trainPriors
        qdaMod.priors = trainPriors
        ldaModself = ldaMod.fit(XrTrain, yTrain)
        qdaMod.fit(XrTrain, yTrain)       
        rfMod.fit(XrTrain, yTrain)        
        
        ldaYes += np.around((ldaMod.score(Xr[test[goodInd]], yGood[test[goodInd]]))*goodInd.size)
        qdaYes += np.around((qdaMod.score(Xr[test[goodInd]], yGood[test[goodInd]]))*goodInd.size)
        rfYes += np.around((rfMod.score(Xr[test[goodInd]], yGood[test[goodInd]]))*goodInd.size)
        cvCount += goodInd.size
                 
    ldaYes = int(ldaYes)
    qdaYes = int(qdaYes)
    rfYes = int(rfYes)
    
    p = 1.0/nClasses
    ldaP = 0
    qdaP = 0
    rfP = 0    
    for k in range(ldaYes, cvCount+1):
        ldaP += binom.pmf(k, cvCount, p)
        
    for k in range(qdaYes, cvCount+1):
        qdaP += binom.pmf(k, cvCount, p)
        
    for k in range(rfYes, cvCount+1):
        rfP += binom.pmf(k, cvCount, p)
        
    print ("Number of classes %d. Chance level %.2f %%" % (nClasses, 100.0/nClasses))
    print ("LDA: %.2f %% (%d/%d p=%.4f)" % (100.0*ldaYes/cvCount, ldaYes, cvCount, ldaP))
    print ("QDA: %.2f %% (%d/%d p=%.4f)" % (100.0*qdaYes/cvCount, qdaYes, cvCount, qdaP))
    print ("RF: %.2f %% (%d/%d p=%.4f)" % (100.0*rfYes/cvCount, rfYes, cvCount, rfP))
   # return ldaYes, qdaYes, rfYes, cvCount, ldaP, qdaP, rfP, nClasses, weights
    return 100.0*ldaYes/cvCount, 100.0*qdaYes/cvCount, 100.0*rfYes/cvCount
