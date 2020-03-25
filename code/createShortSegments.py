from soundsig.sound import temporal_envelope
import os
import numpy as np
import matplotlib.pyplot as plt
from findTimeQuartile import findTimeQuartile
os.chdir("/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/filteredCalls")
# Find all the wave files 
isound = 0   
fs = 44100
segmentLength = 2000
allSegments = []
allMidSegments = []
birdName = []
for fname in os.listdir('.'):
	if fname.endswith('.npy'):
		isound += 1;        
		# Read the sound file        
		sound = np.load(fname)  
		birdname = fname[0:10]
		print ('Processing sound %d:%s' % (isound, fname))
		birdName.append(birdname)
		Quartiletime = findTimeQuartile(sound, fs)
		for i in range(len(Quartiletime)):
			allSegments.append(sound[Quartiletime[i]-int(segmentLength/2) : Quartiletime[i]+int(segmentLength/2)])
		#allMidSegments.append(sound[int(len(sound)/2)-int(segmentLength/2) : int(len(sound)/2)+int(segmentLength/2) ])

#np.save('/Users/Alicia/Desktop/BirdCallProject/dataAnalysis/allMidSegment_Len%d.npy' %segmentLength, allMidSegments)
np.save('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis/allSegment_Len%d.npy' %segmentLength, allSegments)
#np.save('/Users/Alicia/Desktop/BirdCallProject/Freq_20_10000/dataAnalysis/birdName.npy', birdName)
#print(birdName ==birdName1)