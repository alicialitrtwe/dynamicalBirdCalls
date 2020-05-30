
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
def fundEstimator(soundIn, fs, win1, t=None, debugFig = 0, maxFund = 1500, minFund = 300, lowFc = 200, highFc = 6000, minSaliency = 0.5, minFormantFreq = 500, maxFormantBW = 1000, method='Stack'):
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
    #nfilt = 1024
    #if soundLen < 1024:
    #    print('Error in fundEstimator: sound too short for bandpass filtering, len(soundIn)=%d' % soundLen)
    #    return (np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), soundLen)
#
    # high pass filter the signal
    #highpassFilter = firwin(nfilt-1, 2.0*lowFc/fs, pass_zero=False)
    #padlen = min(soundLen-10, 3*len(highpassFilter))
    #soundIn = filtfilt(highpassFilter, [1.0], soundIn, padlen=padlen)
#
    ## low pass filter the signal
    #lowpassFilter = firwin(nfilt, 2.0*highFc/fs)
    #padlen = min(soundLen-10, 3*len(lowpassFilter))
    #soundIn = filtfilt(lowpassFilter, [1.0], soundIn, padlen=padlen)

    # Plot a spectrogram?
    #if debugFig:
    #    plt.figure(9)
    #    (tDebug ,freqDebug ,specDebug , rms) = spectrogram(soundIn, fs, 1000.0, 50, min_freq=0, max_freq=10000, nstd=6, log=True, noise_level_db=50, rectify=True) 
    #    plot_spectrogram(tDebug, freqDebug, specDebug)

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
    fund = np.zeros(nt)
    fund2 = np.zeros(nt)
    sal = np.zeros(nt)
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
        fund[it] = float('nan')
        sal[it] = float('nan')
        fund2[it] = float('nan')
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
    
        # Calculate the auto-correlation
        lags = np.arange(-maxlags, maxlags+1, 1)
        autoCorr = correlation_function(soundWin, soundWin, lags)
        ind0 = int(np.where(lags == 0)[0][0])  # need to find lag zero index
    
        # find peaks
        indPeaksCorr = detect_peaks(autoCorr, mph=autoCorr.max()/10.0)
    
        # Eliminate center peak and all peaks too close to middle    
        indPeaksCorr = np.delete(indPeaksCorr,np.where( (indPeaksCorr-ind0) < fs/maxFund)[0])
        pksCorr = autoCorr[indPeaksCorr]
    
        # Find max peak
        if len(pksCorr)==0:
            pitchSaliency = 0.1               # 0.1 goes with the detection of peaks greater than max/10
        else:
            indIndMax = np.where(pksCorr == max(pksCorr))[0][0]
            indMax = indPeaksCorr[indIndMax]   
            fundCorrGuess = fs/abs(lags[indMax])
            pitchSaliency = autoCorr[indMax]/autoCorr[ind0]

        sal[it] = pitchSaliency
    
        if sal[it] < minSaliency:
            continue

        # Calculate the envelope of the auto-correlation after rectification
        envCorr = temporal_envelope(autoCorr, fs, cutoff_freq=maxFund, resample_rate=None) 
        locsEnvCorr = detect_peaks(envCorr, mph=envCorr.max()/10.0)
        pksEnvCorr = envCorr[locsEnvCorr]
    
        # Find the peak closest to zero
        if locsEnvCorr.size > 1:
            lagdiff = np.abs(locsEnvCorr[0]-ind0)
            indIndEnvMax = 0
                             
            for indtest in range(1,locsEnvCorr.size):
                lagtest = np.abs(locsEnvCorr[indtest]-ind0)
                if lagtest < lagdiff:
                    lagdiff = lagtest
                    indIndEnvMax = indtest                   
          
        # Take the first peak after the one closest to zero
            if indIndEnvMax+2 > len(locsEnvCorr):   # No such peak - use data for correlation function
                fundCorrAmpGuess = fundCorrGuess
                indEnvMax = indMax
            else:
                indEnvMax = locsEnvCorr[indIndEnvMax+1]
                if lags[indEnvMax] == 0 :  # This should not happen
                    print('Error: Max Peak in enveloppe auto-correlation found at zero delay')
                    fundCorrAmpGuess = fundCorrGuess
                    indEnvMax = indMax
                else:
                    fundCorrAmpGuess = fs/lags[indEnvMax]
        else:
            fundCorrAmpGuess = fundCorrGuess
            indEnvMax = indMax

        # Calculate power spectrum and cepstrum
        Y = fft(soundWin, n=winLen+1)
        f = (fs/2.0)*(np.array(range(int((winLen+1)/2+1)), dtype=float)/float((winLen+1)//2))
        fhigh = np.where(f >= highFc)[0][0]
    
        powSound = 20.0*np.log10(np.abs(Y[0:(winLen+1)//2+1]))    # This is the power spectrum
        powSoundGood = powSound[0:fhigh]
        maxPow = max(powSoundGood)
        powSoundGood = powSoundGood - maxPow   # Set zero as the peak amplitude
        powSoundGood[powSoundGood < - 60] = -60    
    
        # Calculate coarse spectral enveloppe
        p = np.polyfit(f[0:fhigh], powSoundGood, 3)
        powAmp = np.polyval(p, f[0:fhigh]) 
    
        # Cepstrum
        CY = dct(powSoundGood-powAmp, norm = 'ortho')            
    
        tCY = 1000.0*np.array(range(len(CY)))/fs          # Units of Cepstrum in ms
        fCY = np.zeros(tCY.size)
        fCY[1:] = 1000.0/tCY[1:] # Corresponding fundamental frequency in Hz.
        fCY[0] = fs*2.0          # Nyquist limit not infinity
        lowInd = np.where(fCY<lowFc)[0]
        if lowInd.size > 0:
            flowCY = np.where(fCY < lowFc)[0][0]
        else:
            flowCY = fCY.size
            
        fhighCY = np.where(fCY < highFc)[0][0]
    
        # Find peak of Cepstrum
        indPk = np.where(CY[fhighCY:flowCY] == max(CY[fhighCY:flowCY]))[0][-1]
        indPk = fhighCY + indPk 
        fmass = 0
        mass = 0
        indTry = indPk
        while (CY[indTry] > 0):
            fmass = fmass + fCY[indTry]*CY[indTry]
            mass = mass + CY[indTry]
            indTry = indTry + 1
            if indTry >= len(CY):
                break

        indTry = indPk - 1
        if (indTry >= 0 ):
            while (CY[indTry] > 0):
                fmass = fmass + fCY[indTry]*CY[indTry]
                mass = mass + CY[indTry]
                indTry = indTry - 1
                if indTry < 0:
                    break

        fGuess = fmass/mass
    
        if (fGuess == 0  or np.isnan(fGuess) or np.isinf(fGuess) ):              # Failure of cepstral method
            fGuess = fundCorrGuess

        fundCepGuess = fGuess
    
        # Force fundamendal to be bounded
        if (fundCepGuess > maxFund ):
            i = 2
            while(fundCepGuess > maxFund):
                fundCepGuess = fGuess/i
                i += 1
        elif (fundCepGuess < minFund):
            i = 2
            while(fundCepGuess < minFund):
                fundCepGuess = fGuess*i
                i += 1
    
        # Fit Gaussian harmonic stack
        maxPow = max(powSoundGood-powAmp)

        # This is the matlab code...
        # fundFitCep = NonLinearModel.fit(f(1:fhigh)', powSoundGood'-powAmp, @synSpect, [fundCepGuess ones(1,9).*log(maxPow)])
        # modelPowCep = synSpect(double(fundFitCep.Coefficients(:,1)), f(1:fhigh))

        vars = np.concatenate(([fundCorrGuess], np.ones(9)*np.log(maxPow)))
        bout = leastsq(residualSyn, vars, args = (f[0:fhigh], powSoundGood-powAmp)) 
        modelPowCep = synSpect(bout[0], f[0:fhigh])
        errCep = sum((powSoundGood - powAmp - modelPowCep)**2)
    
        vars = np.concatenate(([fundCorrGuess*2], np.ones(9)*np.log(maxPow)))
        bout2 = leastsq(residualSyn, vars, args = (f[0:fhigh], powSoundGood-powAmp)) 
        modelPowCep2 = synSpect(bout2[0], f[0:fhigh])
        errCep2 = sum((powSoundGood - powAmp - modelPowCep2)**2)
    
        if errCep2 < errCep:
            bout = bout2
            modelPowCep =  modelPowCep2

        fundStackGuess = bout[0][0]
        if (fundStackGuess > maxFund) or (fundStackGuess < minFund ):
            fundStackGuess = float('nan')

        # Store the result depending on the method chosen
        if method == 'AC':
            fund[it] = fundCorrGuess
        elif method == 'ACA':
            fund[it] = fundCorrAmpGuess
        elif method == 'Cep':
            fund[it] = fundCepGuess
        elif method == 'Stack':
            fund[it] = fundStackGuess  
            
        # A second cepstrum for the second voice
        #     CY2 = dct(powSoundGood-powAmp'- modelPowCep)
    
        if  not np.isnan(fundStackGuess):
            powLeft = powSoundGood- powAmp - modelPowCep
            maxPow2 = max(powLeft)
            f2 = 0
            if ( maxPow2 > maxPow*0.5):    # Possible second peak in central area as indicator of second voice.
                f2 = f[np.where(powLeft == maxPow2)[0][0]]
                if ( f2 > 1000 and f2 < 4000):
                    if (pitchSaliency > minSaliency):
                        fund2[it] = f2

    meanfund = np.mean(fund[~np.isnan(fund)]) if np.size(fund[~np.isnan(fund)])>0 else float("nan")
\
    return meanfund