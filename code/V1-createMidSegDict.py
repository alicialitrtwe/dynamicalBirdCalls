import numpy as np
import os
from soundsig.sound import BioSound 

# Go to the folder that has the h5 files corresponding to the BioSound objects.
frequencyL, frequencyH = [20, 10000]
h5dir = '/Users/Alicia/Desktop/AdultVocalizations/h5files_%s_%s' %(frequencyL, frequencyH)
os.chdir(h5dir)

birdName = []
allMidSeg = []
allFirstSeg = []
allThirdSeg = []
segmentLength = 2000
#dimension = 20
#tau = 4
for fname in os.listdir('.'):
    if fname.endswith('.h5'):
        # Allocate object and read data
        myBioSound = BioSound()
        myBioSound.readh5(fname)
        # These are our two identifier - the emitter (bird) and the call type
        Bird = np.array2string(myBioSound.emitter)[2:-1]
        callType = np.array2string(myBioSound.type)[2:-1]
        sound = myBioSound.sound
        midSeg = sound[int((len(sound)/2))-int(segmentLength/2):int((len(sound)/2))+int(segmentLength/2)]
        firstSeg = sound[int((len(sound)/4))-int(segmentLength/2):int((len(sound)/4))+int(segmentLength/2)]
        thirdSeg = sound[int((len(sound)/4*3))-int(segmentLength/2):int((len(sound)/4*3))+int(segmentLength/2)]
        #segDict.append({"Bird": Bird, "midSeg": midSeg})
        birdName.append(Bird)
        allMidSeg.append(midSeg)
        allFirstSeg.append(firstSeg)
        allThirdSeg.append(thirdSeg)



#np.save('birdName_300_1500.npy',birdName)
allMidSeg = np.array(allMidSeg)
np.save('allMidSeg_%s_%s2000.npy' %(frequencyL, frequencyH), allMidSeg)
allFirstSeg = np.array(allFirstSeg)
np.save('allFirstSeg_%s_%s2000.npy' %(frequencyL, frequencyH), allFirstSeg)
allThirdSeg = np.array(allThirdSeg)
np.save('allThirdSeg_%s_%s2000.npy' %(frequencyL, frequencyH), allThirdSeg)
print('saved')
