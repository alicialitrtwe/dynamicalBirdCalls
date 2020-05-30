import matplotlib.mlab as mlab
import numpy as np
from math import ceil

from scipy.fftpack import fft
from scipy.optimize import leastsq

from soundsig.sound import lpc, temporal_envelope, synSpect, residualSyn
from soundsig.detect_peaks import detect_peaks
from soundsig.signal import correlation_function

def find_fund(segment_list, segment_len, Save=False):
    fund_feat_list = np.array([fund_feat(segment) for segment in segment_list])
    if Save==True:
        np.save('TrainingData/fund_feat_list_%d.npy' %(segment_len), fund_feat_list) 
    return fund_feat_list

def fund_feat(segment, fs=44100, maxFund = 1500, minFund = 300, lowFc = 200, highFc = 6000, minSaliency = 0.5, f_high=10000, minFormantFreq=500, maxFormantBW=1000, windowFormant = 0.1):

# Adapted from soundsig.sound: fundEstimator
# Calculates vowel formant frequencies using linear predictive coding (LPC). 
# The formant frequencies are obtained by finding the roots of the prediction polynomial.
    maxlags = int(2*ceil((float(fs)/minFund)))
    winLen = len(segment)

    A, E, K = lpc(segment, 8)    # 8 degree polynomial
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
        if ( frqsFormants[kk]>minFormantFreq and bw[kk]<maxFormantBW):        
            formants.append(frqsFormants[kk])
    formants = np.array(formants)
    form1 = float('nan')
    form2 = float('nan')
    form3 = float('nan')
    if len(formants) > 0 : 
        form1 = formants[0]
    if len(formants) > 1 : 
        form2 = formants[1]
    if len(formants) > 2 : 
        form3 = formants[2]

    # Calculate the auto-correlation
    lags = np.arange(-maxlags, maxlags+1, 1)
    autoCorr = correlation_function(segment, segment, lags)
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

    sal = pitchSaliency

#   if sal < minSaliency:
#   continue

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
    Y = fft(segment, n=winLen+1)
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
    fund = fundStackGuess 

    return np.array([form1, form2, form3, fund, sal])
