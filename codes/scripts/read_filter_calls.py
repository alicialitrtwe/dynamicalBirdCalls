from soundsig.sound import WavFile
from scipy.signal import filtfilt
import os
import numpy as np
import matplotlib.pyplot as plt
from soundsig.signal import bandpass_filter

os.chdir('/Users/Alicia/ScienceProjects/DynamicalBirdCalls/Data/AdultVocalizations')
if not os.path.exists('filtered_calls'):
    os.makedirs('filtered_calls')
# Find all the wave files 
isound = 0   
low_freq = 250
high_freq = 12000
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
        # bandpass filter the signal
        filtered_call = bandpass_filter(soundIn, sample_rate=fs, low_freq=low_freq, high_freq=high_freq)
        np.save("/Users/Alicia/ScienceProjects/DynamicalBirdCalls/Data/ProcessedData/filtered_calls/%s.npy" %fname[:-4], filtered_call)