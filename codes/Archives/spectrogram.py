import numpy as np
from soundsig.timefreq import gaussian_stft
def spectrogram(s, sample_rate, spec_sample_rate, freq_spacing, min_freq=0, max_freq=None, nstd=2,  cmplx = True):
    """
        Given a sound pressure waveform, s, compute the complex spectrogram. 
        See documentation on gaussian_stft for additional details.
        Returns: 
          t, freq, timefreq, rms
              t: array of time values to use as x axis
              freq: array of frequencies to use as y axis
              timefreq: the spectrogram (a time-frequency represnetaion)
              rms : the time varying average
              
        Arguments:
            REQUIRED:
                s: sound pressssure waveform
                sample_rate: sampling rate for s in Hz
                spec_sample_rate: sampling rate for the output spectrogram in Hz. This variable sets the overlap for the windows in the STFFT.
                freq_spacing: the time-frequency scale for the spectrogram in Hz. This variable determines the width of the gaussian window. 
            
            OPTIONAL            
                complex = False: returns the absolute value
                use min_freq and max_freq to save space
                nstd = number of standard deviations of the gaussian in one window.
    """
     
    # We need units here!!
    increment = 1.0 / spec_sample_rate
    window_length = nstd / (2.0*np.pi*freq_spacing)
    t,freq,timefreq,rms = gaussian_stft(s, sample_rate, window_length, increment, nstd=nstd, min_freq=min_freq, max_freq=max_freq)

    # rms = spec.std(axis=0, ddof=1)
    if cmplx == False:
        timefreq = np.abs(timefreq)
        
    return t, freq, timefreq, rms