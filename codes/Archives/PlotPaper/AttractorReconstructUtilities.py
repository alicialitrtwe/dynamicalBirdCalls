import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.animation as animation
from matplotlib import colors
from scipy.signal import argrelextrema

def TimeDelayReconstruct(TimeSeries, tauIndex, Dimension):
    length = len(TimeSeries) - (Dimension-1)*tauIndex
    ReconstructedAttractor = np.zeros((length, Dimension)) 
    for i in range(0, length):
        ReconstructedAttractor[i] = TimeSeries[i:i+ (Dimension-1)*tauIndex+1:tauIndex]

    return ReconstructedAttractor
        
def TimeSeries3D(TimeSeries, tauIndex, Dimension, Name, axis_limit = None, elev = 90, azim = 0, ax=None):
    ## Generate a 3D graph of a time series embedded in its 3D phase space
    ## Inputs:
    #       TimeSeries: one dimensional array
    #       tauIndex: the selected time delay
    
    length = len(TimeSeries) - (Dimension-1)*tauIndex
    ReconstructedAttractor = np.zeros((length, Dimension)) 
    for i in range(0, length):
        ReconstructedAttractor[i] = TimeSeries[i:i+ (Dimension-1)*tauIndex+1:tauIndex]
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    data = ReconstructedAttractor.T       
    if axis_limit == None:
        axis_limit = np.amax(abs(ReconstructedAttractor))
    ax.set_xlim3d(-axis_limit, axis_limit)
    ax.set_ylim3d(-axis_limit, axis_limit)
    ax.set_zlim3d(-axis_limit, axis_limit)    
    ax.view_init(elev = elev, azim = azim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    color_array = np.arange(data.shape[1])
    ax.scatter(data[0], data[1], data[2], '.', lw=0, s = 25, c=color_array, edgecolors=None, alpha=0.65,  cmap=plt.cm.get_cmap('viridis', 100))
    ax.set_title('%s' %Name)

    
def TimeSeries2Ani(TimeSeries, tauIndex, Dimension, Name, interval = 20, axis_limit = None, colormap = 'viridis', elev = 90, azimspeed = 0.2):
    length = len(TimeSeries) - (Dimension-1)*tauIndex
    ReconstructedAttractor = np.zeros((length, Dimension)) 
    for i in range(0, length):
        ReconstructedAttractor[i] = TimeSeries[i:i+ (Dimension-1)*tauIndex+1:tauIndex]
    pca = PCA(n_components=3)
    ReconstructedAttractor = pca.fit_transform(ReconstructedAttractor) 
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    #ax.axis('off')
    data = ReconstructedAttractor.T       
    if axis_limit == None:
        axis_limit = np.amax(abs(ReconstructedAttractor))
    ax.set_xlim3d(-axis_limit, axis_limit)
    ax.set_ylim3d(-axis_limit, axis_limit)
    ax.set_zlim3d(-axis_limit, axis_limit)
    ax.view_init(elev, 0)
    
    color_array = np.arange(data.shape[1])
    trajectory = ax.scatter([], [], [], '.', lw = 0, s = 25, alpha=0.65, edgecolors=None)
    trajectory.set_cmap("%s" %colormap)
    recent = ax.scatter([], [], [], '.', lw = 0, s = 25, alpha=1, edgecolors=None)
    recent.set_cmap("%s" %colormap)
    def update_trajectory(i, data, trajectory, recent):
        trajectory._offsets3d = data[0:3, :i]
        recent._offsets3d = data[0:3, i-16:i] 
        
        trajectory.set_cmap("%s" %colormap)
        recent.set_cmap("%s" %colormap)
        trajectory.set_array(color_array)
        recent.set_array(color_array)
        ax.set_title('time = %d ms'%i)
        ax.view_init(elev, azimspeed*i)
        return trajectory, recent
    
    # Creating the Animation object
    ani = animation.FuncAnimation(fig, update_trajectory, frames = data.shape[1], fargs=(data, trajectory, recent), interval="%s" %interval, blit=False, repeat=False)
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps = 1000, metadata=dict(artist='Me'), bitrate=3000)
    #ani.save('ani_BDY.mp4', writer=writer)
    plt.show()
    return ani


def MI_plugin(x, y, base = 2):
    bins = np.ceil(np.log2(len(x)))+1
    histgram, x_edges, y_edges = np.histogram2d(x[~np.isnan(x)], y[~np.isnan(y)], bins = bins)
    Pxy = histgram / float(np.sum(histgram))
    Px = np.sum(Pxy, axis=1)  # Get the marginal probability of x by summing over y
    Py = np.sum(Pxy, axis=0)  # Get the marginal probability of y by summing over x
    PxPy = Px[:, None] * Py[None, :]
    non_zero = Pxy > 0 # Only non-zero Pxy terms contribute to the sum
    return np.sum(Pxy[non_zero] * np.log2(Pxy[non_zero] / PxPy[non_zero]))


def MI_tau(time_series, code=MI_plugin,tauIndex=20):
    # Input functions: 
    #     time_series: shape (n_samples, n_features). n_samples is the number of points in the time series, 
    #     and n_features is the dimension of the parameter space.array_like, shape (n,m). The time series to be unfolded
    #     code: MI_knn_1D or MI_knn_multiD. For one dimensional data, both codes give same result, but MI_knn_1D runs much faster, 
    #     becuase it uses KDTree to find nearest neighbor.  
    #     tauIndex: the number of time delays for which to estimate mutual information
    # Output functions:
    #     mi: mi(tauIndex) is the estimated mutual information between the time series at [0,t_total - tauIndex] and [tauIndex, t_total]
    N = len(time_series)
    mi = np.zeros(tauIndex)  
    #f = FloatProgress(min=0, max=tauIndex) # instantiate the progress bar
    #display(f) # display the bar
    for i in range(0, tauIndex): 
        mi[i] = code(time_series[:N-1-i],time_series[i+1:])
    #  f.value += 1
    return mi

def MI_lcmin(TimeSeries, order=1, PLOT=False):
    # Find the indice for the first and the second local minimum
    # Inputs:
    #     data: array (n,)
    #     order: how many points on each side to use for the comparison 
    # Output:
    #     lcmin: indice for the first local minimum
    MI = MI_tau(TimeSeries)
    lcmin = argrelextrema(MI, np.less, order=order)[0][0]+1
    if PLOT == True:
        plt.figure()
        plt.plot(MI)
        plt.show()
    return lcmin 

def TimeDelayReconstruct_allD(TimeSeries, tauIndex, Dimension = 10):
    # Inputs:
    #     TimeSeries: time series, (n, 1) narray  
    #     lcmin_index: the index for the time delay selected
    #     length: the number of TimeSeries points to use for reconstruction. See Abarbanel P. 49. Oversampling needs to be avoided.
    #     samplerate: take 1 TimeSeries point for every 'samplerate' TimeSeries points when reconstructing the state space
    #     Dimension: do the reconstruction for [1, Dimension] dimensions
    # Output: 
    #     ReconstructedAttractors_allD: a list of length (Dimension-1) containing points in the reconstructed [2, Dimension] dimensional state space
    length = len(TimeSeries) - (Dimension-1)*tauIndex
    ReconstructedAttractors_allD = []
    for d in range(1, Dimension+1):
        ReconstructedAttractor_D = np.zeros((length, d)) 
        for i in range(0, length):
            ReconstructedAttractor_D[i] = TimeSeries[i:i+(d-1)*tauIndex+1:tauIndex]
        ReconstructedAttractors_allD.append(ReconstructedAttractor_D)
    return ReconstructedAttractors_allD

def fnn_Cao(ReconstructedAttractors_allD, PLOT=False):
    # Find the percentage of false nearst neighbor in dimension d 
    # Inputs: 
    #    ReconstructedAttractors_allD: vectors in reconstructed state space
    #    Dimension: do the calculation for [1, Dimension-2] dimensions. 
    #           The max dimension we can get to here has to be two less than Dimension for the previous step  
    #           because we have to go one dimension up to define false/truth nearest neighbors, and one more up to find E1, E2 
    # Output: the percentage of false nearest neighbor in dimensions [1, Dimension]-2

    # find the nearest neighbor points in dimension [1, Dimension], 
    # and calculate the Euclidian distance between them and the original points
    Dimension = len(ReconstructedAttractors_allD)
    E = []
    Es = []
    E1 = []
    E2 = []
    for d in range(1, Dimension):
        ReconstructedAttractor_D = ReconstructedAttractors_allD[d-1] 
        ReconstructedAttractor_Dplus1 = ReconstructedAttractors_allD[d]
        tree = ss.cKDTree(ReconstructedAttractor_D)
        dd, ii = tree.query(ReconstructedAttractor_D, k = [2], p=float('inf'))
        ii = np.squeeze(ii)
        dd = np.squeeze(dd)
        Reconstru_D_NN = ReconstructedAttractor_D[ii]  # find the nearest neighbors 
        Reconstru_D_NN = np.vstack(Reconstru_D_NN)
        Reconstru_Dplus1_NN = ReconstructedAttractor_Dplus1[ii]
        Es_d = np.mean(abs(ReconstructedAttractor_Dplus1[:,-1] - Reconstru_Dplus1_NN[:,-1]))
        R_d_plus1 = np.amax(abs(ReconstructedAttractor_Dplus1 - Reconstru_Dplus1_NN), axis = 1) # Maximum norm  
        a_i_d = R_d_plus1/(dd+1e-15)
        E_d = np.mean(a_i_d)
        E.append(E_d)
        Es.append(Es_d)
    for d in range(1, Dimension-1):
        E1_d = E[d]/E[d-1]
        E2_d = Es[d]/Es[d-1]
        E1.append(E1_d)
        E2.append(E2_d)
    if PLOT == True:
        plt.figure()
        plt.plot(range(1,len(E1)+1), E1, 'bo-', label=r'$E_1(d)$')
        plt.plot(range(1,len(E1)+1), E2, 'go-', label=r'$E_2(d)$')
        plt.title(r'AFN for time series')
        plt.xlabel(r'Embedding dimension $d$')
        plt.ylabel(r'$E_1(d)$ and $E_2(d)$')
        plt.legend()
        plt.show()
    return [np.array(E1).T, np.array(E2).T]