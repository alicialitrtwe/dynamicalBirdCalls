from soundsig.sound import WavFile, plot_spectrogram, spectrogram
from scipy.signal import firwin, filtfilt
from soundsig.signal import lowpass_filter
import os
from scipy.signal import argrelextrema
import shutil
import numpy as np
import matplotlib.pyplot as plt
os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_20_10000/filteredCalls')
# Find all the wave files 
isound = 250   

fs = 44100
for fname in os.listdir('.'):
    if fname.endswith('.npy') and isound<300:
        isound += 1;        
        # Read the sound file        
        sound = np.load(fname)    
        print ('Processing sound %d:%s' % (isound, fname))
        soundLen = len(sound)
        sound = sound - sound.mean()
        sound_env = lowpass_filter(np.abs(sound), float(fs), 20.0) 
        minimum = argrelextrema(sound_env, np.less, order=2)[0]
        minimum = minimum[np.where(sound_env[minimum] < 0.1*max(sound_env))]
        minimum = minimum[np.where((1000< minimum)&(minimum<soundLen-1000))]
        print(minimum)
#       if len(minimum) == 1:
#           seg1 = sound[:minimum[0]]
#           seg2 = sound[minimum[0]:]
#           print(len(seg1) > 1323, len(seg2) > 1323)
#           if len(seg1) > 1323 and len(seg2) > 1323:
#               print('true')
#               np.save("%sSeg1.npy" %fname[:-4], seg1)
#               np.save("%sSeg2.npy" %fname[:-4], seg2)
#           else: 
#               print('short seg')
#           shutil.move('/Users/Alicia/Desktop/BirdCallProject/Freq_20_10000/filteredCalls/%s' %fname, \
#               '/Users/Alicia/Desktop/BirdCallProject/Freq_20_10000/reSegmentedCalls/%s' %fname)
#       if len(minimum) == 2:
#           seg1 = sound[:minimum[0]]
#           seg2 = sound[minimum[0]:minimum[1]]
#           seg3 = sound[minimum[1]:]    
#           print(len(seg1), len(seg2), len(seg3))    
#           if len(seg1) > 1323 and len(seg2) > 1323 and len(seg3) > 1323:
#               np.save("%sSeg1.npy" %fname[:-4], seg1)
#               np.save("%sSeg2.npy" %fname[:-4], seg2)
#               np.save("%sSeg3.npy" %fname[:-4], seg3)
#           else: 
#               print('short seg')
#           shutil.move('/Users/Alicia/Desktop/BirdCallProject/Freq_20_10000/filteredCalls/%s' %fname, \
#               '/Users/Alicia/Desktop/BirdCallProject/Freq_20_10000/reSegmentedCalls/%s' %fname)

#        plt.plot(sound_env/sound_env.max(), color="red", linewidth=2)
        plt.figure()
        (tDebug ,freqDebug ,specDebug , rms) = spectrogram(sound, fs, 1000.0, 50, min_freq=0, max_freq=10000, nstd=6, cmplx=True) 
        plot_spectrogram(tDebug, freqDebug, specDebug)
        plt.show()