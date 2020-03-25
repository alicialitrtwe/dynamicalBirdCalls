
from soundsig.signal import lowpass_filter, gaussian_window, correlation_function
from soundsig.timefreq import gaussian_stft
from soundsig.detect_peaks import detect_peaks
from soundsig.sound import temporal_envelope, residualSyn, synSpect
from math import ceil
import numpy as np
from scipy.io.wavfile import read as read_wavfile
from scipy.fftpack import fft, ifft, fftfreq, fft2, ifft2, dct
from scipy.signal import resample, firwin, filtfilt
from scipy.linalg import inv, toeplitz
from scipy.optimize import leastsq

def lpc(signal, order):
    """Compute the Linear Prediction Coefficients.
    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:
      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]
    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.
    Parameters
    ----------
    signal: array_like
        input signal   
    order : int
        LPC order (the output will have order + 1 items)"""

    order = int(order)

    if signal.ndim > 1:
        raise ValueError("Array of rank > 1 not supported yet")
    if order > signal.size:
        raise ValueError("Input signal must have a lenght >= lpc order")

    if order > 0:
        p = order + 1
        r = np.zeros(p, signal.dtype)
        # Number of non zero values in autocorrelation one needs for p LPC
        # coefficients
        nx = np.min([p, signal.size])
        x = np.correlate(signal, signal, 'full')
        r[:nx] = x[signal.size-1:signal.size+order]
        phi = np.dot(inv(toeplitz(r[:-1])), -r[1:])
        return np.concatenate(([1.], phi)), None, None
    else:
        return np.ones(1, dtype = signal.dtype), None, None
def formantEst(soundIn, fs, win1, t=None, debugFig = 0, maxFund = 1500, minFund = 300, lowFc = 200, highFc = 6000, minSaliency = 0.5, minFormantFreq = 500, maxFormantBW = 1000, method='Stack'):
    """
    Estimates the fundamental frequency of a complex sound.
    soundIn is the sound pressure waveformlog spectrogram.
    fs is the sampling rate
    t is a vector of time values in s at which the fundamental will be estimated.
    The sound must include at least 1024 sample points
    The optional parameter with defaults are
    Some user parameters (should be part of the function at some time)
       debugFig = 0         Set to zero to eliminate figures.
       maxFund = 1500       Maximum fundamental frequency
       minFund = 300        Minimum fundamental frequency
       lowFc = 200          Low frequency cut-off for band-passing the signal prior to auto-correlation.
       highFc = 6000        High frequency cut-off
       minSaliency = 0.5    Threshold in the auto-correlation for minimum saliency - returns NaN for pitch values is saliency is below this number
    Four methods are available: 
    'AC' - Peak of the auto-correlation function
    'ACA' - Peak of envelope of auto-correlation function 
    'Cep' - First peak in cepstrum 
    'Stack' - Fitting of harmonic stacks (default - works well for zebra finches)
    
    Returns
           sal     - the time varying pitch saliency - a number between 0 and 1 corresponding to relative size of the first auto-correlation peak
           fund     - the time-varying fundamental in Hz at the same resolution as the spectrogram.
           fund2   - a second peak in the spectrum - not a multiple of the fundamental a sign of a second voice
           form1   - the first formant, if it exists
           form2   - the second formant, if it exists
           form3   - the third formant, if it exists
           soundLen - length of sal, fund, fund2, form1, form2, form3
    """

    # Band-pass filtering signal prior to auto-correlation
    soundLen = len(soundIn)

    # Initializations and useful variables
    soundLen = len(soundIn)
    sound_dur = soundLen / fs
    
    if t is None:
        # initialize t to be spaced by  1 ms increments if not specified     
        _si = 1e-3
        npts = int(sound_dur / _si)
        t = np.arange(npts) * _si

    nt=len(t)
    soundRMS = np.zeros(nt)
    form1 = np.zeros(nt)
    form2 = np.zeros(nt)
    form3 = np.zeros(nt)

    #  Calculate the size of the window for the auto-correlation
    alpha = 5                          # Number of sd in the Gaussian window
    winLen = int(np.fix((2.0*alpha/minFund)*fs))  # Length of Gaussian window based on minFund
    if (winLen%2 == 0):  # Make a symmetric window
        winLen += 1
        
    winLen2 = 2**12+1   # This looks like a good size for LPC - 4097 points
    gt, w = gaussian_window(winLen, alpha)
    gt2, w2 = gaussian_window(winLen2, alpha)
    maxlags = int(2*ceil((float(fs)/minFund)))

    # First calculate the rms in each window
    for it in range(nt):
        tval = t[it]               # Center of window in time
        if tval >= sound_dur:
            continue
        tind = int(np.fix(tval*fs))    # Center of window in ind
        tstart = tind - (winLen-1)//2
        tend = tind + (winLen-1)//2
    
        if tstart < 0:
            winstart = - tstart
            tstart = 0
        else:
            winstart = 0
        
        if tend >= soundLen:
            windend = winLen - (tend-soundLen+1) - 1
            tend = soundLen-1
        else:
            windend = winLen-1
            
        soundWin = soundIn[tstart:tend]*w[winstart:windend]
        soundRMS[it] = np.std(soundWin)
    
    soundRMSMax = max(soundRMS)

    # Calculate the auto-correlation in windowed segments and obtain 4 guess values of the fundamental
    # fundCorrGuess - guess from the auto-correlation function
    # fundCorrAmpGuess - guess form the amplitude of the auto-correlation function
    # fundCepGuess - guess from the cepstrum
    # fundStackGuess - guess taken from a fit of the power spectrum with a harmonic stack, using the fundCepGuess as a starting point
    #  Current version use fundStackGuess as the best estimate...

    soundlen = 0
    for it in range(nt):
        form1[it] = float('nan')
        form2[it] = float('nan')
        form3[it] = float('nan')
        
        if (soundRMS[it] < soundRMSMax*0.1):
            continue
    
        soundlen += 1
        tval = t[it]               # Center of window in time
        if tval >= sound_dur:       # This should not happen here because the RMS should be zero
            continue
        tind = int(np.fix(tval*fs))    # Center of window in ind
        tstart = tind - (winLen-1)//2
        tend = tind + (winLen-1)//2
    
        if tstart < 0:
            winstart = - tstart
            tstart = 0
        else:
            winstart = 0
        
        if tend >= soundLen:
            windend = winLen - (tend-soundLen+1) - 1
            tend = soundLen-1
        else:
            windend = winLen-1

            (winLen2-1)//2 - tind
            - (winLen2-1)//2 - tind + 4093 + soundLen-1 - 1

        tstart2 = tind - (winLen2-1)//2
        tend2 = tind + (winLen2-1)//2
    
        if tstart2 < 0:
            winstart2 = - tstart2
            tstart2 = 0
        else:
            winstart2 = 0
        
        if tend2 >= soundLen:
            windend2 = winLen2 - (tend2-soundLen+1) - 1
            tend2 = soundLen-1
        else:
            windend2 = winLen2-1
            
        soundWin = soundIn[tstart:tend]*w[winstart:windend]


        if win1 == False:      
            soundWin2 = soundIn[tstart2:tend2]*w2[winstart2:windend2]
        else:
            soundWin2 = soundIn
        # Apply LPC to get time-varying formants and one additional guess for the fundamental frequency
        # TODO (kevin): replace this with librosa 
        A, E, K = lpc(soundWin2, 8)    # 8 degree polynomial
        rts = np.roots(A)          # Find the roots of A
        rts = rts[np.imag(rts)>=0]  # Keep only half of them
        angz = np.arctan2(np.imag(rts),np.real(rts))
    
        # Calculate the frequencies and the bandwidth of the formants
        frqsFormants = angz*(fs/(2*np.pi))
        indices = np.argsort(frqsFormants)
        bw = -0.5*(fs/(2*np.pi))*np.log(np.abs(rts))  # FIXME (kevin): I think this line was broken before... it was using 1/2
    
        # Keep formants above 500 Hz and with bandwidth < 500 # This was 1000 for bird calls
        formants = []
        for kk in indices:
            if ( frqsFormants[kk]>minFormantFreq and bw[kk] < maxFormantBW):        
                formants.append(frqsFormants[kk])
        formants = np.array(formants) 
        
        if len(formants) > 0 : 
            form1[it] = formants[0]
        if len(formants) > 1 : 
            form2[it] = formants[1]
        if len(formants) > 2 : 
            form3[it] = formants[2]
    # Fix formants.
    meanf1 = np.mean(form1[~np.isnan(form1)])
    meanf2 = np.mean(form2[~np.isnan(form2)])
    meanf3 = np.mean(form3[~np.isnan(form3)])

    for it in range(nt):
        if ~np.isnan(form1[it]):
            df11 = np.abs(form1[it]-meanf1)
            df12 = np.abs(form1[it]-meanf2)
            df13 = np.abs(form1[it]-meanf3)
            if df12 < df11:
                if df13 < df12:
                    if ~np.isnan(form3[it]):
                        df33 = np.abs(form3[it]-meanf3)
                        if df13 < df33:
                            form3[it] = form1[it]
                    else:
                      form3[it] = form1[it]
                else:
                    if ~np.isnan(form2[it]):
                        df22 = np.abs(form2[it]-meanf2)
                        if df12 < df22:
                            form2[it] = form1[it]
                    else:
                        form2[it] = form1[it]
                form1[it] = float('nan')
            if ~np.isnan(form2[it]):  
                df21 = np.abs(form2[it]-meanf1)
                df22 = np.abs(form2[it]-meanf2)
                df23 = np.abs(form2[it]-meanf3)
                if df21 < df22 :
                    if ~np.isnan(form1[it]):
                        df11 = np.abs(form1[it]-meanf1)
                        if df21 < df11:
                            form1[it] = form2[it]
                    else:
                      form1[it] = form2[it]
                    form2[it] = float('nan')
                elif df23 < df22:
                    if ~np.isnan(form3[it]):
                        df33 = np.abs(form3[it]-meanf3)
                        if df23 < df33:
                            form3[it] = form2[it]
                    else:
                        form3[it] = form2[it]
                    form2[it] = float('nan')
            if ~np.isnan(form3[it]):
                df31 = np.abs(form3[it]-meanf1)
                df32 = np.abs(form3[it]-meanf2)
                df33 = np.abs(form3[it]-meanf3)
                if df32 < df33:
                    if df31 < df32:
                        if ~np.isnan(form1[it]):
                            df11 = np.abs(form1[it]-meanf1)
                            if df31 < df11:
                                form1[it] = form3[it]
                        else:
                            form1[it] = form3[it]
                    else:
                        if ~np.isnan(form2[it]):
                            df22 = np.abs(form2[it]-meanf2)
                            if df32 < df22:
                                form2[it] = form3[it]
                        else:
                            form2[it] = form3[it]
                    form3[it] = float('nan')

    meanf1 = np.mean(form1[~np.isnan(form1)]) if np.size(form1[~np.isnan(form1)])>0 else float("nan")
    meanf2 = np.mean(form2[~np.isnan(form2)]) if np.size(form1[~np.isnan(form2)])>0 else float("nan")
    meanf3 = np.mean(form3[~np.isnan(form3)]) if np.size(form1[~np.isnan(form3)])>0 else float("nan")

    return meanf1, meanf2, meanf3