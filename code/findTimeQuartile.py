from soundsig.sound import temporal_envelope
import numpy as np
import matplotlib.pyplot as plt
import os

#os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/filteredCalls')
def findTimeQuartile(sound, fs):
	#tdata = np.arange(0, len(sound), 1)
	amp  = temporal_envelope(sound, fs, cutoff_freq=20)
    # Here are the parameters
	ampdata = amp/np.sum(amp)
	cumAmp = np.array([np.sum(ampdata[:i]) for i in range(len(ampdata))])
	
	Q1time = np.where(np.abs(cumAmp-0.25) == np.min(np.abs(cumAmp-0.25)))[0][0]
	#Q2time = int(np.sum(tdata*ampdata))
	Q2time = np.where(np.abs(cumAmp-0.5) == np.min(np.abs(cumAmp-0.5)))[0][0]
	Q3time = np.where(np.abs(cumAmp-0.75) == np.min(np.abs(cumAmp-0.75)))[0][0]
	return Q1time, Q2time, Q3time
	#plt.figure()
	#plt.plot(ampdata,'b')
	#plt.axvline(x = Q1time)
	#plt.axvline(x = Q2time)
	#plt.axvline(x = Q3time)
	#plt.show()
	#return Q1time, Q2time, Q3time
#isound = 0
#for fname in os.listdir('.'):
#	if fname.endswith('.npy') and isound<= 8:
#		isound += 1;        
#		# Read the sound file        
#		sound = np.load(fname) 
#		findTimeQuartile(sound, 44100)
#