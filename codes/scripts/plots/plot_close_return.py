import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
import os
from matplotlib.ticker import MultipleLocator, FixedLocator
import seaborn as sns

from dynamicalbirdcalls.config import processed_data_dir
from dynamicalbirdcalls.set_plot_params import plot_params

os.chdir(processed_data_dir)
plot_params()

# Parameters
n_samples, n_features = 100, 144

# dataset
X = np.load('filtered_calls/LblBla4548_130418-DC-46.npy')
X = X[2900:4000].reshape(1, -1) #3200:3700

# Recurrence plot transformation
#X = np.load('filtered_calls/RedRas3600_110615-DC-10.npy')
#X = X[1500:3000].reshape(1, -1) #3200:3700
rp = RecurrencePlot(dimension=1, time_delay=1,
					threshold='distance',	
                     percentage=5)
n = 250
X_rp = rp.fit_transform(X)
X_new = X_rp[0]
Y = np.zeros((len(X_new)-n*2, n))
for i in range(len(X_new)-n*2):
	for j in range(n):
		Y[i, j]= X_new[i+n, i+n+j]
fig = plt.figure(figsize=(5,2.5))
ax = fig.add_subplot(1, 1, 1) 
ax.imshow(Y.T, cmap='binary', origin='lower', extent=[0,len(X_new)-n*2,0,n])
color=sns.color_palette("Set1", n_colors=3, desat=0.7)[0]
ax.plot([0, 330], [50, 50], c=color)
ax.plot([0, 330], [70, 70], c=color)
ax.plot([0, len(X_new)-n*2], [110, 110], c=color)
ax.plot([0, len(X_new)-n*2], [130, 130], c=color)
ax.plot([len(X_new)-n*2, len(X_new)-n*2], [110, 130], c=color)
ax.plot([0,0], [110, 130], c=color)
ax.plot([0, 0], [50, 70], c=color)
ax.plot([330, 330], [50, 70], c=color)

ax.xaxis.set_tick_params(direction='in', width=0.4)
ax.yaxis.set_tick_params(direction='in', width=0.4)
ax.yaxis.set_major_locator(MultipleLocator(60))
ax.xaxis.set_major_locator(MultipleLocator(200))
ax.set_xlabel('t (ms)')
ax.set_ylabel('Period')
ax.set_yticklabels(['T', '0', 'T', '2T', '3T', '4T'])
ax.set_xticklabels([0, 0, 4.5, 9.0, 13.6])
plt.show()

# Show the results for the first time series
#plt.figure(figsize=(8, 8))
#plt.imshow(X_rp[0], cmap='binary', origin='lower')
#plt.show()
#fig.savefig('/Users/Alicia/AnacondaProjects/CurrentProject/PlotPaper/Fig5CloseReturnPlot.svg', dpi=300, bbox_inches='tight', transparent=True)