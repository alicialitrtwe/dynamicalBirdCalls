import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys
import seaborn as sns
sys.path.append('/Users/Alicia/AnacondaProjects/CurrentProject')
from AttractorReconstructUtilities import TimeSeries3D, TimeDelayReconstruct, MI_lcmin
from sciPlot import updateParams
from matplotlib.ticker import MultipleLocator
updateParams()

freG = 0.1
Fs = 41
Fs = 82
dt = 1/Fs
np.random.seed(11)
t = np.arange(0,20,dt)
x1 = np.zeros(t.shape)

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


#plt.plot(t, np.array(x).T)
#plt.show()

#MI = MI_lcmin(x1, order=5, PLOT=True, tauNumber=200)
#print(MI)
Attractor = []
for xi in x:
	Attractor.append(TimeDelayReconstruct(xi, 102, 3))
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1, projection='3d') 
ax.tick_params(labelsize = 8)

ax.set_xlim3d(-1.2, 1.2)
ax.set_ylim3d(-1.2, 1.2)
ax.set_zlim3d(-1.2, 1.2)    

#ax.set_xticklabels([])
#ax.set_yticklabels([])
#ax.set_zticklabels([])
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
ax.w_xaxis.line.set_color("lightgrey")
ax.w_yaxis.line.set_color("lightgrey")
ax.w_zaxis.line.set_color("lightgrey")
#color_array = np.arange(Attractor.shape[0])

ax.view_init(elev=13, azim=150)
for i in range(len(Attractor)):
	Attractori = Attractor[i]
	ax.scatter(Attractori[:,0], Attractori[:,1], Attractori[:,2], '.', c=sns.color_palette("Set1", n_colors=3, desat=0.76)[i], lw=0, s=8, edgecolors=None, alpha=0.95)
ax.set_xlabel(r'$X(t)$' "\n")
ax.set_ylabel(r'$X(t-\tau)$' "\n" )
ax.set_zlabel(r'$X(t-2\tau)$' "\n"  )


ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.zaxis.set_major_locator(MultipleLocator(1))
fig.tight_layout()


plt.show()#
#fig.savefig('Fig2c timbreReconstruct.svg', dpi=300, bbox_inches='tight', transparent=True)