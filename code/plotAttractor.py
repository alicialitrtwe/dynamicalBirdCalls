import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import colors
from AttractorReconstructUtilities import TimeDelayReconstruct, TimeSeries3D

TenSongs = np.load('/Users/Alicia/Desktop/SongAttractor/Attractor3D/TenSongs.npy')
TenSongs = TenSongs[()]

signal1 = TenSongs['BDY'][40500:44500]
plt.close('all')
TimeSeries3D(signal1, 2, 3, 'BDY', axis_limit = None, elev = 90, azim = 0)
plt.show()