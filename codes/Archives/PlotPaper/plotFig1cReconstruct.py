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
from AttractorReconstructUtilities import TimeSeries3D, TimeDelayReconstruct, MI_lcmin
from sciPlot import updateParams
updateParams()
# Define ODEs for Lorenz Attractor
def lorenz_ode(state, t, parameters):
    x, y, z = state
    sigma, beta, rho = parameters
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

state0 = [0., 1., 1.05]
params = [10.0, 8/3.0, 28.0]
t = np.linspace(0, 50, 5000)          
Attractor = odeint(lorenz_ode, state0, t, args=(params,))[2000:]
x = Attractor[:,0]
MI = MI_lcmin(x, order=1, PLOT=False)

Attractor = TimeDelayReconstruct(x, MI, 5)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1, projection='3d') 

axis_limit = np.amax(abs(Attractor))


ax.set_xlim3d(-25, 25)
ax.set_ylim3d(-25, 25)
ax.set_zlim3d(-25, 18)    
#ax.view_init(elev = 0, azim = azim)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_xlabel('X\n\n\n')
ax.set_ylabel('Y\n\n\n')

ax.zaxis.set_rotate_label(True) 
ax.set_zlabel('z\n\n\n\n', rotation=90)
ax.grid(False)
ax.xaxis._axinfo['tick']['inward_factor'] = 0
ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
ax.yaxis._axinfo['tick']['inward_factor'] = 0
ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
ax.zaxis._axinfo['tick']['inward_factor'] = 0
ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
ax.w_yaxis.set_pane_color((0.87, 0.87, 0.87, 0.87))
ax.w_xaxis.set_pane_color((0.92, 0.92, 0.92, 0.92))
ax.w_zaxis.set_pane_color((0.96, 0.96, 0.96, 0.96))
ax.xaxis.pane.set_edgecolor("lightgrey")
ax.yaxis.pane.set_edgecolor("lightgrey")
ax.zaxis.pane.set_edgecolor("lightgrey")
#[t.set_va('center') for t in ax.get_yticklabels()]
#[t.set_ha('left') for t in ax.get_yticklabels()]
#[t.set_va('center') for t in ax.get_xticklabels()]
#[t.set_ha('right') for t in ax.get_xticklabels()]
#[t.set_va('center') for t in ax.get_zticklabels()]
#[t.set_ha('left') for t in ax.get_zticklabels()]
ax.w_xaxis.line.set_color("lightgrey")
ax.w_yaxis.line.set_color("lightgrey")
ax.w_zaxis.line.set_color("lightgrey")
color_array = np.arange(Attractor.shape[0])
ax.view_init(elev=13, azim=150)
ax.scatter(Attractor[:,0], Attractor[:,1], Attractor[:,2], '.', lw=0, s = 8, c=color_array, edgecolors=None, alpha=0.65,  cmap=plt.cm.get_cmap('viridis', 1000))
ax.scatter(Attractor[:,0], Attractor[:,1], np.zeros(len(Attractor))-25-1, '.', lw=0, s = 8, c=color_array, edgecolors=None, alpha=0.20,  cmap=plt.cm.get_cmap('viridis', 1000))
#ax.scatter(Attractor[:,0], Attractor[:,1], Attractor[:,2], '.', lw=0, s = 2, c='k', edgecolors=None, alpha=0.6)
ax.set_xlabel(r'$X(t)$' "\n" "\n")
ax.set_ylabel(r'$X(t-\tau)$' "\n" "\n")
ax.set_zlabel(r'$X(t-2\tau)$' "\n" "\n" "\n" "\n")
fig.tight_layout()

plt.show()#
fig.savefig('Fig1c LorenzReconstruct.svg', dpi=300, bbox_inches='tight', transparent=True)