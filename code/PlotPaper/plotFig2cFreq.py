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
fig = plt.figure(figsize = (5, 3))
ax = fig.add_subplot(1, 1, 1)
#for k in np.arange(1,9,2):
#	x1 = x1 + np.sin(2*np.pi*k*t*freG)/(k**(5/4))
freq = [1,3,5,7]	
amp = [1, 1, 1, 1]
for i in range(len(freq)):
	plt.plot([freq[i], freq[i]], [0, amp[i]], lw=2, c='k')

ax.xaxis.set_tick_params(top='off', direction='in', width=0.4)
ax.yaxis.set_tick_params(right='off', direction='in', width=0.4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Frequency')
ax.set_ylabel('Magnitude')

ax.set_xlim(0, 7.5)
ax.set_ylim(0, 2)

ax.xaxis.set_major_locator(FixedLocator([0, 1, 3, 5, 7]))
ax.yaxis.set_major_locator(MultipleLocator(1))

ax.set_xticklabels(['0', 'f', '3f', '5f', '7f'])
plt.show()
fig.savefig('Fig2c timbreFreq.svg', dpi=300, bbox_inches='tight', transparent=True)