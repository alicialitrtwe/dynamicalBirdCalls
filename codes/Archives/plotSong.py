import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import colors
from detect_peaks import detect_peaks
from scipy.signal import resample
from utilities import wav2data
from AttractorReconstructUtilities import TimeDelayReconstruct, TimeSeries3D, TimeSeries2Ani, MI_lcmin
from soundsig.sound import WavFile, plot_spectrogram, spectrogram


#TenSongs = np.load('/Users/Alicia/Desktop/SongAttractor/Attractor3D/TenSongs.npy')
#TenSongs = TenSongs[()]
zebra = np.array(wav2data('/Users/Alicia/Desktop/SongData/Simple.wav')[0][0])

call = np.load('/Users/Alicia/Desktop/BirdCallProject/Freq_20_10000/filteredCalls/WhiBlu5698_110329-DC-14.npy')
#print(MI_lcmin(zebra[17300:17800], PLOT=True))
#TimeSeries2Ani(zebra[17500:17700], 3, 10, 'zebra', axis_limit = None, elev = 60, azimspeed=0.3)
#TimeSeries3D(zebra[17500:17700], 3, 5, 'zebra', axis_limit = None, elev = 60)
plt.figure()
#(tDebug ,freqDebug ,specDebug , rms) = spectrogram(call[5000:6000], 44100, 1000.0, 50, min_freq=0, max_freq=10000, nstd=6, cmplx=True) 
#plot_spectrogram(tDebug, freqDebug, specDebug)
plt.figure()
plt.plot(call[2050:2660])
plt.show()