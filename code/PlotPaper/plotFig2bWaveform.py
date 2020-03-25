import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys
import seaborn as sns
from matplotlib import rc, rcParams
#sys.path.append('/Users/Alicia/AnacondaProjects/CurrentProject')
from AttractorReconstructUtilities import TimeSeries3D, TimeDelayReconstruct, MI_lcmin
from sciPlot import updateParams
from matplotlib.ticker import MultipleLocator, FixedLocator


updateParams()
freG = 0.1
Fs = 41
dt = 1/Fs
np.random.seed(11)
t = np.arange(0,30,dt)
x1 = np.zeros(t.shape)
fig = plt.figure(figsize = (5, 6))
ax = fig.subplots(3, 1, sharex='col')
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=3)
for k in np.arange(1,9,2):
	x1 = x1 + np.sin(2*np.pi*k*t*freG)/(k**1)
x2 = np.zeros(t.shape)
for k in np.arange(1,9,2):
	x2 = x2+ np.sin(2*np.pi*k*t*freG+ np.pi/4*((-1)**((k-1)/2)+1) )/(k**1)
	#x2 = x2+ np.sin(2*np.pi*k*t*freG + np.pi/4*(k-1))/(k**1)
x3 = np.zeros(t.shape)
for k in np.arange(1,9,2):
	x3 = x3+ np.sin(2*np.pi*k*t*freG + np.random.randn(1)*np.pi)/(k**1)

x = [x1, x2, x3]


#ax[0].plot(t, x,'black')
#ax[0].set_ylabel(r'$X(t)$')
#ax[1].plot(t+100, y,color=colorName[2])
#ax[1].set_ylabel(r'$X(t-\tau)$')
#ax[2].plot(t+200, z, color=colorName[10])
#ax[2].set_ylabel(r'$X(t-2\tau)$')
for i in range(len(x)):
	ax[i].plot(t, np.array(x[i]).T, c=sns.color_palette("Set1", n_colors=3, desat=0.7)[i])
	#
	ax[i].xaxis.set_tick_params(top='off', direction='in', width=0.4)
	ax[i].yaxis.set_tick_params(right='off', direction='in', width=0.4)
	ax[i].spines['top'].set_visible(False)
	ax[i].spines['right'].set_visible(False)
	ax[i].yaxis.set_major_locator(MultipleLocator(1))
	ax[i].set_xlim(0, 30)
	ax[i].set_ylim(-1.8, 1.8)
#plt.legend()

ax[2].xaxis.set_major_locator(MultipleLocator(10))
ax[2].set_xlabel('t')
ax[1].set_ylabel('Amplitude')
ax[2].set_xticklabels(['T', '0', 'T', '2T', '3T'])

plt.show()
fig.savefig('Fig2b timbreWaveform.svg', dpi=300, bbox_inches='tight', transparent=True)