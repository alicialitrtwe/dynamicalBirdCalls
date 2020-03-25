import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interactive
from ipywidgets import FloatProgress
from IPython.display import display
import scipy.spatial as ss
import scipy.stats as sst
from scipy.special import digamma,gamma
from math import log,pi,exp
from scipy.integrate import odeint
from AttractorReconstructUtilities import TimeSeries3D, TimeDelayReconstruct
from matplotlib import style
from sciPlot import updateParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, FixedLocator
from soundsig.sound import WavFile, plot_spectrogram, spectrogram
import seaborn as sns
updateParams()

TenSongs = np.load('/Users/Alicia/Desktop/SongAttractor/Attractor3D/TenSongs.npy', allow_pickle=True)
TenSongs = TenSongs[()]
#Attractor = TimeDelayReconstruct(TenSongs['BDY'][37300:48000], 2, 3)


wavForm = TenSongs['BDY'][37300:48000]
wavForm = wavForm[2200:2350]
t = np.arange(0, len(wavForm))
fig, ax = plt.subplots(figsize=(5, 2)) 
ax.plot(t, wavForm, c=sns.color_palette("Set1", n_colors=3, desat=0.8)[1]) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.xaxis.set_tick_params(top='off', direction='out', width=1)
ax.yaxis.set_tick_params(right='off', direction='out', width=1)

ax.set_xlabel('t (s)')
ax.set_ylabel('Amplitude')
ax.yaxis.set_major_locator(MultipleLocator(15000))
ax.xaxis.set_major_locator(MultipleLocator(50))

ax.set_xticklabels([0, 0.050, 0.051, 0.052, 0.053])
#ax.set_yticks(yticks)

plt.show()
fig.savefig('Fig3BDYSWavFormFig3c.svg', dpi=300, bbox_inches='tight', transparent=True)
