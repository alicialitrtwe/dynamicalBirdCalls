from soundsig.sound import BioSound 
from soundsig.sound import WavFile
from soundsig.signal import bandpass_filter
import os
import numpy as np
os.chdir('/Users/Alicia/Desktop/AdultVocalizations')
normalize = False
# Find all the wave files 
isound = 0   
for fname in os.listdir('.'):
    if fname.endswith('.wav') and 'DC' in fname:
        isound += 1;
        
        # Read the sound file
        print ('Processing sound %d:%s' % (isound, fname))
        soundIn = WavFile(file_name=fname) 
        filename, file_extension = os.path.splitext(fname)
        
        # Here we parse the filename to get the birdname and the call type. 
        # You will have to write custom code to extract your own identifiers.
        birdname = filename[0:10]
        calltype = filename[18:20]     
        
        
        # Normalize if wanted
        if normalize :
            maxAmp = np.abs(soundIn.data).max() 
        else :
            maxAmp = 1.0
    
    # Create BioSound Object
        myBioSound = BioSound(soundWave=soundIn.data.astype(float)/maxAmp, fs=float(soundIn.sample_rate), emitter=birdname, calltype = calltype)
        myBioSound.sound = bandpass_filter(myBioSound.sound, myBioSound.samprate, 300,1500, filter_order=4)
        if 'DC' in np.array2string(myBioSound.type)[2:-1]:
            print(myBioSound.type)
            # Save the results
            fh5name = 'h5files_300_1500/%s.h5' % (filename)
            myBioSound.saveh5(fh5name)
        else:
            pass