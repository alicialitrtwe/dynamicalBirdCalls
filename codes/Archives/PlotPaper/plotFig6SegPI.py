import numpy as np
import os
from sklearn.decomposition import PCA
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
from matplotlib.ticker import MultipleLocator, FixedLocator
updateParams()
import sys
sys.path.append('/Users/Alicia/AnacondaProjects/CurrentProject')
from generatePersImage import generatePD, generatePI
from timeDelayPCA import timeDelayPCA

allSegments = np.load('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis/allSegment_Len500.npy')
birdName = np.load('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis/birdName.npy')
birdName = np.repeat(birdName, 3)
norm = 'Normed2Max' 
PCA_n_componentsL = [0] 
pixelsL = [15]
spreadL = [1]
norm = 'Normed2Max'
segmentLengthL = [500]
embDimL = [25]
tauL = [3]
PCA_n_components = 0 
pixels = 15
spread = 1
savePD = False 
savePI = False
#
for Len in range(len(segmentLengthL)):
	for Dim in range(len(embDimL)):
		for Tau in range(len(tauL)):
			for PCA in range(len(PCA_n_componentsL)):

					segmentLength = segmentLengthL[Len]
					embDim = embDimL[Dim]
					tau = tauL[Tau]
					PCA_n_components = PCA_n_componentsL[PCA]
					if PCA_n_components < embDim and embDim*tau < 300:
						                                                                    
						#allTimeDelayedSeg = timeDelayPCA(allSegments, segmentLength, embDim, tau, PCA_n_components)
						#allPDs = generatePD(allTimeDelayedSeg, segmentLength, embDim, tau, PCA_n_components, savePD, norm)
			
						for p in range(len(pixelsL)):
							for s in range(len(spreadL)):
								pixels = pixelsL[p]
								spread = spreadL[s]
								#if 'MidPD_Len%s_Dim%s_Tau%s_PCA%s.npy' %(segmentLength, embDim, tau, PCA_n_components)
								print('processing segmentLength', segmentLength, 'embDim ', embDim, ' tau ', tau, ' PCA_n_components ', PCA_n_components, ' pixels ', pixels, ' spread ', spread, '\n\n') 
#								#PI0 = generatePI(allPDs, segmentLength, embDim, tau, PCA_n_components, pixels, spread, savePI, norm)
#allPIs = np.load('PI_Len%s_Dim%s_Tau%s_PCA%s_p%s_s%s_%s.npy' %(segmentLength, embDim, tau, PCA_n_components, pixels, spread, norm))

x = allSegments[:4]
allTimeDelayedSeg = timeDelayPCA(x, segmentLength, embDim, tau, PCA_n_components)

from persim import PersImage, plot_diagrams
from ripser import Rips

rips = Rips(maxdim=1, coeff=2)
diagrams_h1 = [rips.fit_transform(data)[1] for data in allTimeDelayedSeg]
pim = PersImage(pixels=[15,15], spread=1)
imgs = pim.transform(diagrams_h1)


fig = plt.figure(figsize=(5,5))
ax = fig.subplots()
ax.imshow(imgs[0], cmap=plt.get_cmap("viridis"))
ax.axis("off")

#pim.show(imgs[0], ax=ax)
#plot_diagrams(diagrams_h1[3], ax=ax, legend=False)

#ax.xaxis.set_major_locator(MultipleLocator(5000))
#ax.yaxis.set_major_locator(MultipleLocator(5000))
##MI = MI_lcmin(x, order=1, PLOT=True)
##print(MI)
#Attractor = TimeDelayReconstruct(x, 2, 3)
#fig = plt.figure(figsize=(5,5))
#ax = fig.add_subplot(1, 1, 1, projection='3d')
#axis_limit = np.amax(abs(Attractor))
#
#ax.set_xlim3d(-axis_limit, axis_limit)
#ax.set_ylim3d(-axis_limit, axis_limit)
#ax.set_zlim3d(-axis_limit, axis_limit)    
#ax.set_xticklabels([])
#ax.set_yticklabels([])
#ax.set_zticklabels([])
#ax.set_xlabel('X\n\n\n')
#ax.set_ylabel('Y\n\n\n')
#ax.zaxis.set_rotate_label(True) 
#ax.set_zlabel('z\n\n\n\n', rotation=90)
#ax.grid(False)
#ax.xaxis._axinfo['tick']['inward_factor'] = 0
#ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
#ax.yaxis._axinfo['tick']['inward_factor'] = 0
#ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
#ax.zaxis._axinfo['tick']['inward_factor'] = 0
#ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
#ax.w_yaxis.set_pane_color((0.87, 0.87, 0.87, 0.87))
#ax.w_xaxis.set_pane_color((0.92, 0.92, 0.92, 0.92))
#ax.w_zaxis.set_pane_color((0.96, 0.96, 0.96, 0.96))
#ax.xaxis.pane.set_edgecolor("lightgrey")
#ax.yaxis.pane.set_edgecolor("lightgrey")
#ax.zaxis.pane.set_edgecolor("lightgrey")
#ax.w_xaxis.line.set_color("lightgrey")
#ax.w_yaxis.line.set_color("lightgrey")
#ax.w_zaxis.line.set_color("lightgrey")
#color_array = np.arange(Attractor.shape[0])
#ax.view_init(elev=13, azim=150)
#ax.scatter(Attractor[:,0], Attractor[:,1], Attractor[:,2], '.', lw=0, s = 8, c=color_array, edgecolors=None, alpha=0.65,  cmap=plt.cm.get_cmap('viridis', 100))
##ax.scatter(Attractor[:,0], Attractor[:,1], np.zeros(len(Attractor))-25-1, '.', lw=0, s = 8, c=color_array, edgecolors=None, alpha=0.20,  cmap=plt.cm.get_cmap('viridis', 1000))
##ax.scatter(Attractor[:,0], Attractor[:,1], Attractor[:,2], '.', lw=0, s = 2, c='k', edgecolors=None, alpha=0.6)
#ax.set_xlabel(r'$X(t)$' "\n" "\n")
#ax.set_ylabel(r'$X(t-\tau)$' "\n" "\n")
#ax.set_zlabel(r'$X(t-2\tau)$' "\n" "\n" "\n" "\n")
fig.tight_layout()
plt.show()#
fig.savefig('Fig6SegPI0.svg', dpi=300, bbox_inches='tight', transparent=True)
