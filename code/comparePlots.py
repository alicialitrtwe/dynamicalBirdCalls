from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import os

frequencyL, frequencyH = [300, 1500]
# Go to the folder that has the h5 files corresponding to the BioSound objects.
h5dir = '/Users/Alicia/Desktop/AdultVocalizations/h5files_%s_%s' %(frequencyL, frequencyH)
os.chdir(h5dir)

birdName = np.load('birdName_%s_%s.npy' %(frequencyL, frequencyH))
allFirstSeg = np.load('allFirstSeg_%s_%s.npy' %(frequencyL, frequencyH))
allMidSeg = np.load('allMidSeg_%s_%s.npy' %(frequencyL, frequencyH))
allThirdSeg = np.load('allThirdSeg_%s_%s.npy' %(frequencyL, frequencyH))

enc = LabelEncoder()
label_encoder = enc.fit(birdName)
y = label_encoder.transform(birdName) + 1

os.chdir('/Users/Alicia/AnacondaProjects/CurrentProject')
segmentLength = 400
embDim = 20
tau = 3
PCA_n_components = 3
from timeDelayPCA import timeDelayPCA
allMidTDSeg = timeDelayPCA(allMidSeg, segmentLength, embDim, tau, PCA_n_components)
allFirstTDSeg = timeDelayPCA(allFirstSeg, segmentLength, embDim, tau, PCA_n_components)
allThirdTDSeg = timeDelayPCA(allThirdSeg, segmentLength, embDim, tau, PCA_n_components)
fig = plt.figure(figsize = (8.5, 8))
#for i in range(len(allTimeDelayedSeg)):

label = [7,7]
iplot = 1
for i in range(0, 8): 
    ax = plt.subplot(3,8,iplot, projection='3d')
    #ax = plt.subplot(2,8,iplot)
    data3D_1 = allFirstTDSeg
    ax.scatter(data3D_1[np.where(y==label[0])[0][i],:,0], data3D_1[np.where(y==label[0])[0][i],:,1], data3D_1[np.where(y==label[0])[0][i], :,2])
    #ax.plot(allMidSeg_20_2000[np.where(y==label[0])[0][i], 900:1100])
    print(birdName[np.where(y==label[0])[0][i]])
    
    ax = plt.subplot(3,8,iplot+8, projection='3d')
    data3D_2 = allMidTDSeg
    ax.scatter(data3D_2[np.where(y==label[1])[0][i], :,0], data3D_2[np.where(y==label[1])[0][i], :,1], data3D_2[np.where(y==label[1])[0][i], :,2])
    #ax.plot(allMidSeg_20_2000[np.where(y==label[1])[0][i], 900:1100])
    print(birdName[np.where(y==label[1])[0][i]])

    ax = plt.subplot(3,8,iplot+16, projection='3d')
    data3D_3 = allThirdTDSeg
    ax.scatter(data3D_3[np.where(y==label[1])[0][i], :,0], data3D_3[np.where(y==label[1])[0][i], :,1], data3D_3[np.where(y==label[1])[0][i], :,2])
    #ax.plot(allMidSeg_20_2000[np.where(y==label[1])[0][i], 900:1100])
    print(birdName[np.where(y==label[1])[0][i]])

    iplot+=1
plt.show()
#rips.plot(diagrams_h1[indices[i]], show=False)


#classes, classesCount = np.unique(y, return_counts = True)  # Classes to be discriminated should be same as ldaMod.classes_
#goodIndClasses = np.array([n >= 10 for n in classesCount])
#goodInd = np.array([b in classes[goodIndClasses] for b in y])
#yGood = y[goodInd]
#XGood = X[goodInd]        
#classes, classesCount = np.unique(yGood, return_counts = True) 
#nClasses = classes.size       