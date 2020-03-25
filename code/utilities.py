import wave
import struct
import numpy as np 
import matplotlib.pyplot as plt


def wav2data(wave_file):
    """Given a file-like object or file path representing a wave file,
    decompose it into its constituent PCM data streams.

    Input: A file like object or file path
    Output: A list of lists of integers representing the PCM coded data stream channels
        and the sample rate of the channels (mixed rate channels not supported)
    """
    stream = wave.open(wave_file,"rb")

    num_channels = stream.getnchannels()
    sample_rate = stream.getframerate()
    sample_width = stream.getsampwidth()
    num_frames = stream.getnframes()

    raw_data = stream.readframes( num_frames ) # Returns byte data
    stream.close()

    total_samples = num_frames * num_channels

    if sample_width == 1: 
        fmt = "%iB" % total_samples # read unsigned chars
    elif sample_width == 2:
        fmt = "%ih" % total_samples # read signed 2 byte shorts
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    integer_data = struct.unpack(fmt, raw_data)
    del raw_data # Keep memory tidy (who knows how big it might be)

    channels = [ [] for time in range(num_channels) ]

    for index, value in enumerate(integer_data):
        bucket = index % num_channels
        channels[bucket].append(value)

    return channels, sample_rate
from scipy import signal

def spectrogram(x, nsample, win, step, plot=True):
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
    return f, t, Sxx





