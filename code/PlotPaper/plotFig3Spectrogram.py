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
updateParams()

TenSongs = np.load('/Users/Alicia/Desktop/SongAttractor/Attractor3D/TenSongs.npy', allow_pickle=True)
TenSongs = TenSongs[()]
#Attractor = TimeDelayReconstruct(TenSongs['BDY'][37300:48000], 2, 3)
(tDebug ,freqDebug ,specDebug , rms) = spectrogram(TenSongs['BDY'][37300:48000], 44100, 2000.0, 50, min_freq=0, max_freq=22050, nstd=6, cmplx=True) 
fig, ax =plt.subplots(figsize=(5,2))
plot_spectrogram(tDebug, freqDebug, specDebug, ax=ax, colorbar=False)
ax.set_xlabel('t (s)')
plt.show()
#fig.savefig('Fig3BDYSpectrogram.svg', dpi=300, bbox_inches='tight', transparent=True)
