import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal
def spectrogramScipy(x, nsample, win, step, plot=False):
    f, t, Sxx = signal.spectrogram(x, nsample, window='hanning',
                                  nperseg = win, noverlap= win - step,
                                      detrend=False, scaling='spectrum')

    f = np.array(f)
    Sxx = np.array(Sxx)
    t = np.array(t)

    if plot: 
        fig, ax = plt.subplots(figsize=(4.8, 2.4))
        ax.pcolormesh(t, f, 10 *np.log10(Sxx), cmap='viridis')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]');
        plt.show()
    return t, f, Sxx
