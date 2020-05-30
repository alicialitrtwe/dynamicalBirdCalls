import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import colors
from detect_peaks import detect_peaks
from scipy.signal import resample
from AttractorReconstructUtilities import TimeDelayReconstruct, TimeSeries3D
TenSongs = np.load('/Users/Alicia/Desktop/SongAttractor/Attractor3D/TenSongs.npy')
TenSongs = TenSongs[()]

signal1 = TenSongs['BDY'][41500:42500]
s = signal1/np.max(signal1)
signal2 = TenSongs['BDY'][42500:43500] 
s2 = signal2/np.max(signal2)

Fs = 44100  # sampling frequency
dt = 1/Fs  # sampling interval
t = np.arange(0, len(s)/Fs, dt)

from scipy.signal import argrelextrema
import cmath

result = np.fft.rfft(s)
freqs = np.fft.fftfreq(s.size, 1/Fs)
mag = np.abs(result)
angle = np.angle(result)
np.random.shuffle(angle)
FFT = mag * np.exp(1j*angle)
#FFT = result*cmath.rect(1., 0)
#print(result, cmath.rect(1.,0))
s_re = np.fft.irfft(FFT)
print(s_re)
plt.figure()
plt.plot(s_re)
plt.plot(s, '--')
#TimeSeries3D(np.fft.ifft(FFT)[200:500], 2, 3, 'BDY', axis_limit = None, elev = 90, azim = 0)
plt.show()

s2 = s_re
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))

# plot time signal:
axes[0, 0].set_title("Signal")
axes[0, 0].plot(t, s, color='C0')
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")

# plot different spectrum types:
axes[1, 0].set_title("Magnitude Spectrum")
axes[1, 0].magnitude_spectrum(s, Fs=Fs, color='C1')
axes[1, 0].magnitude_spectrum(s2, Fs=Fs, color='C2')

axes[1, 1].set_title("Log. Magnitude Spectrum")
axes[1, 1].magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')
axes[1, 1].magnitude_spectrum(s2, Fs=Fs, scale='dB', color='C2')

axes[2, 0].set_title("Phase Spectrum ")
axes[2, 0].phase_spectrum(s, Fs=Fs, color='C1')
axes[2, 0].phase_spectrum(s2, Fs=Fs, color='C2')

axes[2, 1].set_title("Angle Spectrum")
axes[2, 1].angle_spectrum(s, Fs=Fs, color='C1')
axes[2, 1].angle_spectrum(s2, Fs=Fs, color='C2')
axes[0, 1].remove()  # don't display empty ax

fig.tight_layout()
plt.show()