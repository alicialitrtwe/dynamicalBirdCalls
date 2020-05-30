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
updateParams()

TenSongs = np.load('/Users/Alicia/Desktop/SongAttractor/Attractor3D/TenSongs.npy', allow_pickle=True)
TenSongs = TenSongs[()]
Attractor = TimeDelayReconstruct(TenSongs['BDY'][37300:48000], 2, 3)


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1, projection='3d') 

axis_limit = np.amax(abs(Attractor)) + 50
ax.set_xlim3d(-axis_limit, axis_limit)
ax.set_ylim3d(-axis_limit, axis_limit)
ax.set_zlim3d(-axis_limit, axis_limit)    
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_xlabel('X\n\n\n')
ax.set_ylabel('Y\n\n\n')

ax.zaxis.set_rotate_label(True) 
ax.set_zlabel('z\n\n\n\n', rotation=90)
ax.grid(False)


ax.w_yaxis.set_pane_color((0.87, 0.87, 0.87, 0.87))
ax.w_xaxis.set_pane_color((0.92, 0.92, 0.92, 0.92))
ax.w_zaxis.set_pane_color((0.96, 0.96, 0.96, 0.96))
ax.xaxis.pane.set_edgecolor("lightgrey")
ax.yaxis.pane.set_edgecolor("lightgrey")
ax.zaxis.pane.set_edgecolor("lightgrey")
ax.w_xaxis.line.set_color("lightgrey")
ax.w_yaxis.line.set_color("lightgrey")
ax.w_zaxis.line.set_color("lightgrey")
color_array = np.arange(Attractor.shape[0])

sc = ax.scatter(Attractor[:,0], Attractor[:,1], Attractor[:,2], '.', lw=0, s = 4, c=color_array, edgecolors=None, alpha=0.75,  cmap=plt.cm.get_cmap('viridis', 1000))
cbar = plt.colorbar(sc, pad=0.8, shrink=.55, aspect=15, ticks=[0, len(Attractor)-1])
cbar.set_ticklabels(['0 s','0.242 s'])
ax.set_xlabel(r'$X(t)$')
ax.set_ylabel(r'$X(t-\tau)$' )
ax.set_zlabel(r'$X(t-2\tau)$' )
ax.xaxis.set_major_locator(MultipleLocator(15000))
ax.yaxis.set_major_locator(MultipleLocator(15000))
ax.zaxis.set_major_locator(MultipleLocator(15000))
ax.set_xticklabels([-15000, 0 , 15000])
ax.set_yticklabels([-15000, 0 , 15000])
ax.set_zticklabels([-15000, 0 , 15000])
ax.xaxis._axinfo['tick']['inward_factor'] = 0
ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
ax.yaxis._axinfo['tick']['inward_factor'] = 0
ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
ax.zaxis._axinfo['tick']['inward_factor'] = 0
ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
#ax.xaxis.set_ticks_position('top')
ax.view_init(elev = 45, azim = 135)
fig.tight_layout()
plt.show()#
##angle 1 elev = 10, azim = 100
## elev=10, azim=10)
fig.savefig('Fig3BDYColor.png', dpi=300, bbox_inches='tight', transparent=True)