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
from spectrogramScipy import spectrogramScipy
from scipy.interpolate import splprep, splev
from AttractorReconstructUtilities import TimeDelayReconstruct
 
updateParams()
zebra = WavFile(file_name='/Users/Alicia/Desktop/SongData/Simple.wav').data.astype(float)
canary = WavFile(file_name='/Users/Alicia/Desktop/SongData/CD2_10x.wav').data.astype(float)[65000:260000]
BDY = np.loadtxt('/Users/Alicia/Desktop/SongData/Porg_BDY.dat')
#Seg = [zebra[17000:17500], canary[-3600:-3100]]
Seg = [zebra[27500:28000], BDY[42500:43000]]
from AttractorReconstructUtilities import MI_lcmin
allMIs = [MI_lcmin(Seg0, order=2, PLOT=False) for Seg0 in Seg]
print(allMIs)
fig = plt.figure(figsize=(5,8))
#for i in range(1, 3):
#	ax = fig.add_subplot(2, 2, 2*i)
##(tDebug ,freqDebug ,specDebug) = spectrogramScipy(Seg[0], 44100, 256, 50)
#	(tDebug ,freqDebug ,specDebug, rms) = spectrogram(Seg[i-1], 44100, 2000, 100, min_freq=0, max_freq=22050, nstd=6, cmplx=True)  
#	plot_spectrogram(tDebug, freqDebug, specDebug, ax=ax, colorbar=True, dBNoise=None)
#	ax.set_yticklabels([0,5,10,15,20])
#	ax.set_ylabel('Frequency (kHz)')
#	ax.xaxis.set_tick_params(direction='in', width=0.4)
#	ax.yaxis.set_tick_params(direction='in', width=0.4)
#	if i == 1:
#		ax.set_xlabel('')
#		ax.set_xticks([])   
#		ax.set_xticklabels([])
#	else:
#		ax.set_xlabel('t (s)')
SinglePeriod = zebra[27558:27623]
#plt.plot(SinglePeriod)
Attractor = TimeDelayReconstruct(SinglePeriod, 2, 3)
tck, u = splprep(Attractor.T, u=None, s=0, per=1)
u_new = np.linspace(u.min(), u.max(), 300)
newpoints = splev(u_new, tck, der=0)
newpoints = np.vstack((newpoints[0], newpoints[1], newpoints[2])).T 
#plt.figure()
#plt.plot(attractor[:,0], attractor[:,1], 'o')
#plt.plot(newpoints[:,0], newpoints[:,1], '-')

MI = [2,2]
axis_limitL = [3350, 17500]
markL = [3000, 15000]
for i in range(1):
	ax = fig.add_subplot(2, 1, i+1, projection='3d')
	#Attractor = TimeDelayReconstruct(SinglePeriod, MI[i], 3) #pure tone
	axis_limit = np.amax(abs(Attractor))
	axis_limit = axis_limitL[i]
	mark = markL[i]
	ax.set_xlim3d(-axis_limit, axis_limit)
	ax.set_ylim3d(-axis_limit, axis_limit)
	ax.set_zlim3d(-axis_limit, axis_limit)    
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_zticklabels([])
	ax.grid(False)
	ax.w_yaxis.set_pane_color((0.92, 0.92, 0.92, 0.92))
	ax.w_zaxis.set_pane_color((0.96, 0.96, 0.96, 0.96))
	ax.xaxis.pane.set_edgecolor("lightgrey")
	ax.yaxis.pane.set_edgecolor("lightgrey")
	ax.zaxis.pane.set_edgecolor("lightgrey")
	ax.w_xaxis.line.set_color("lightgrey")
	ax.w_yaxis.line.set_color("lightgrey")
	ax.w_zaxis.line.set_color("lightgrey")
	color_array = np.arange(Attractor.shape[0])

	sc = ax.scatter(Attractor[:,0], Attractor[:,1], Attractor[:,2], '.', lw=0, s = 20, c=color_array,  alpha=1,  cmap=plt.cm.get_cmap('viridis', 10))
	ax.plot(newpoints[:,0], newpoints[:,1], newpoints[:,2], lw=1, alpha=0.8, c='grey')
	#cbar = plt.colorbar(sc, pad=0.8, shrink=.55, aspect=15, ticks=[0, len(Attractor)-1])
	#cbar.set_ticklabels(['0 s','0.242 s'])
	ax.set_xlabel(r'$X(t)$')
	ax.set_ylabel(r'$X(t-\tau)$' )
	ax.set_zlabel(r'$X(t-2\tau)$')
	ax.xaxis.set_major_locator(MultipleLocator(mark))
	ax.yaxis.set_major_locator(MultipleLocator(mark))
	ax.zaxis.set_major_locator(MultipleLocator(mark))
	ax.set_xticklabels([-mark, 0 , mark])
	ax.set_yticklabels([-mark, 0 , mark])
	ax.set_zticklabels([-mark, 0 , mark])
	ax.xaxis._axinfo['tick']['inward_factor'] = 0
	ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
	ax.yaxis._axinfo['tick']['inward_factor'] = 0
	ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
	ax.zaxis._axinfo['tick']['inward_factor'] = 0
	ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
	#ax.xaxis.set_ticks_position('top')
	ax.view_init(elev = 20, azim = -45)
	ax.dist = 10
	#plt.tight_layout()
fig.subplots_adjust(left=0.15, bottom=None, right=None, top=None, wspace=None, hspace=0.1)
plt.show()
#fig.savefig('Fig4PureToneReconstruct.svg', dpi=300, bbox_inches='tight', transparent=True)
#fig.savefig('Fig4HarmonicReconstructZebra.svg', dpi=300,  transparent=True)