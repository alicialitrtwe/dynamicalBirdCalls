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
import os 
from sciPlot import updateParams
updateParams()


def lorenz_ode(state, t, parameters):
    x, y, z = state
    sigma, beta, rho = parameters
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

state0 = [0., 1., 1.05]
params = [10.0, 8/3.0, 28.0]
t = np.linspace(0, 50, 5000)          
Attractor = odeint(lorenz_ode, state0, t, args=(params,))[2000:]
x_org = Attractor[:,0]



x = x_org[500:-600]
t = np.arange(len(x))
y = x_org[600:-500]
z = x_org[700:-400]
os.chdir('/Users/Alicia/AnacondaProjects/CurrentProject/PlotPaper')
colorName = np.load('colorName.npy')
fig = plt.figure(figsize=(5, 3))

ax = fig.subplots(3, 1, sharex='col')
ax[0].plot(t, x,'black')
ax[0].set_ylabel(r'$X(t)$')
ax[0].axvline(ls='--', c='k', x=len(x),  alpha=0.85, lw=1)
ax[0].axvline(ls='--', c='k', x=len(x)+100,  alpha=0.85, lw=1)
ax[1].plot(t+100, y,color=colorName[2])
ax[1].set_ylabel(r'$X(t-\tau)$')
ax[1].axvline(ls='--', c='k', x=100,  alpha=0.85, lw=1)
ax[1].axvline(ls='--', c='k', x=len(x)+100,  alpha=0.85, lw=1)
ax[2].plot(t+200, z, color=colorName[10])
ax[2].axvline(ls='--', c='k', x=100,  alpha=0.85, lw=1)
ax[2].axvline(ls='--', c='k', x=200,  alpha=0.85, lw=1)
#ax[2].axvline(x=200, color='grey', alpha=0.8, lw=3)
#ax[2].axvline(x=len(x)+100, color='grey', alpha=0.8, lw=3)
ax[2].set_ylabel(r'$X(t-2\tau)$')

ax[2].set_xlabel(r'$t$')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_yticks([])
ax[2].set_yticks([])

ax[0].set_xlim(0, len(x)+200)
ax[1].set_xlim(0, len(x)+200)
ax[2].set_xlim(0, len(x)+200)


plt.show()
fig.savefig('Fig1cTimeDelay.svg', dpi=300, bbox_inches='tight', transparent=True)