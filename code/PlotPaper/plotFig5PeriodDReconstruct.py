import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from sciPlot import updateParams
from matplotlib.ticker import MultipleLocator, FixedLocator
import seaborn as sns
from AttractorReconstructUtilities import TimeSeries3D, TimeDelayReconstruct
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, FixedLocator
from scipy.interpolate import splprep, splev
from AttractorReconstructUtilities import TimeDelayReconstruct
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('/Users/Alicia/AnacondaProjects/CurrentProject')
from detect_peaks import detect_peaks
updateParams()
# Parameters
n_samples, n_features = 100, 144

# Toy dataset
X = np.load('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/filteredCalls/LblBla4548_130418-DC-46.npy')
X = X[2900:4000]
#plt.plot(X)
#462  521  580  638
ind = detect_peaks(X, mph=0.25, mpd=44.1, show=False, edge='both', kpsh=1, valley=1)
print(ind)

fig = plt.figure(figsize=(5,5))
SinglePeriod = X[521:642]
#SinglePeriod = X[462:525]
#plt.plot(SinglePeriod)
Attractor = TimeDelayReconstruct(SinglePeriod, 2, 3)
tck, u = splprep(Attractor.T, u=None, s=0, per=1)
u_new = np.linspace(u.min(), u.max(), 300)
newpoints = splev(u_new, tck, der=0)
newpoints = np.vstack((newpoints[0], newpoints[1], newpoints[2])).T 

axis_limitL = [7800]
markL = [7000]
for i in range(1):
	ax = fig.add_subplot(1, 1, i+1, projection='3d')
	#SinglePeriod = X[462:526]
	Attractor = TimeDelayReconstruct(SinglePeriod, 2, 3) #pure tone
	axis_limit = np.amax(abs(Attractor))
	print(axis_limit)
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
#fig.savefig('Fig5AfterBifur.svg', dpi=300,  transparent=True)