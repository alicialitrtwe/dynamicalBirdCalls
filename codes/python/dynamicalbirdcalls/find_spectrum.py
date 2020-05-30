import matplotlib.mlab as mlab
import numpy as np

def find_spectrum(segment_list, segment_len, Save=False):
    spectrum_list = np.array([spectrum(segment) for segment in segment_list])
    if Save==True:
        np.save('TrainingData/spectrum_list_%d.npy' %(segment_len), spectrum_list)  
    return spectrum_list

def spectrum(segment, fs=44100, f_high=10000):
# Calculates power spectrum and features from power spectrum

# Need to add argument for window size
# f_high is the upper bound of the frequency for saving power spectrum
# nwindow = (1000.0*np.size(soundIn)/samprate)/window_len
# 
    Pxx, Freqs = mlab.psd(segment, Fs=fs, NFFT=1024, noverlap=512)

    # Find quartile power
    cum_power = np.cumsum(Pxx)
    tot_power = np.sum(Pxx)
    quartile_freq = np.zeros(3, dtype = 'int')
    quartile_values = [0.25, 0.5, 0.75]
    nfreqs = np.size(cum_power)
    iq = 0
    for ifreq in range(nfreqs):
        if (cum_power[ifreq] > quartile_values[iq]*tot_power):
            quartile_freq[iq] = ifreq
            iq = iq+1
            if (iq > 2):
                break
             
    # Find skewness, kurtosis and entropy for power spectrum below f_high
    ind_fmax = np.where(Freqs > f_high)[0][0]

    # Description of spectral shape
    spectdata = Pxx[0:ind_fmax]
    freqdata = Freqs[0:ind_fmax]
    spectdata = spectdata/np.sum(spectdata)
    meanspect = np.sum(freqdata*spectdata)
    stdspect = np.sqrt(np.sum(spectdata*((freqdata-meanspect)**2)))
    skewspect = np.sum(spectdata*(freqdata-meanspect)**3)
    skewspect = skewspect/(stdspect**3)
    kurtosisspect = np.sum(spectdata*(freqdata-meanspect)**4)
    kurtosisspect = kurtosisspect/(stdspect**4)
    entropyspect = -np.sum(spectdata*np.log2(spectdata))/np.log2(ind_fmax)
        # Storing the values       
    q1 = Freqs[quartile_freq[0]]
    q2 = Freqs[quartile_freq[1]]
    q3 = Freqs[quartile_freq[2]]
    return np.array([q1, q2, q3, meanspect, stdspect, skewspect, kurtosisspect, entropyspect])
