from soundsig.sound import WavFile
from scipy.signal import firwin, filtfilt
import os
import numpy as np
import matplotlib.pyplot as plt
from soundsig.signal import bandpass_filter
os.chdir('/Users/Alicia/Desktop/BirdCallProject/AdultVocalizations')
normalize = False
# Find all the wave files 
isound = 0   
lowFc = 20
highFc = 10000
nfilt = 1024
fs = 44100
for fname in os.listdir('.'):
    if fname.endswith('.wav') and 'DC' in fname:
        isound += 1;        
        # Read the sound file        
        soundIn = WavFile(file_name=fname).data.astype(float)
        filename, file_extension = os.path.splitext(fname)
        birdname = filename[0:10]
        calltype = filename[18:20]     
        print ('Processing sound %d:%s, %s' % (isound, fname, calltype))
        
        # Normalize if wanted
        #if normalize :
        #    maxAmp = np.abs(soundIn.data).max() 
        #else :
        #    maxAmp = 1.0

        #soundIn.data.astype(float)/maxAmp

        soundLen = len(soundIn)
        highpassFilter = firwin(nfilt-1, 2.0*lowFc/fs, pass_zero=False)
        padlen = min(soundLen-10, 3*len(highpassFilter))
        hpresult = filtfilt(highpassFilter, [1.0], soundIn, padlen=padlen)
        # low pass filter the signal
        lowpassFilter = firwin(nfilt, 2.0*highFc/fs)
        padlen = min(soundLen-10, 3*len(lowpassFilter))
        lpresult = filtfilt(lowpassFilter, [1.0], hpresult, padlen=padlen)

        #bpresult = bandpass_filter(soundIn, fs, lowFc, highFc, filter_order=5, rescale=False)
        #plt.figure()
        #plt.plot(soundIn[1000:2000])
        #plt.plot(bpresult[1000:2000], '-.')
        #plt.show()
        np.save("/Users/Alicia/Desktop/BirdCallProject/Freq_20_10000/filteredCalls/%s.npy" %fname[:-4], lpresult)
