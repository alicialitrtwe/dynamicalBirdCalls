from soundsig.sound import temporal_envelope
import numpy as np
import matplotlib.pyplot as plt


def find_time_quartile(sound, fs):
	amp  = temporal_envelope(sound, fs, cutoff_freq=20)
	amp_normed = amp/np.sum(amp)
	amp_cumul = np.array([np.sum(amp_normed[:i]) for i in range(len(amp_normed))])
	q1_time = np.where(np.abs(amp_cumul-0.25) == np.min(np.abs(amp_cumul-0.25)))[0][0]
	q2_time = np.where(np.abs(amp_cumul-0.50) == np.min(np.abs(amp_cumul-0.50)))[0][0]
	q3_time = np.where(np.abs(amp_cumul-0.75) == np.min(np.abs(amp_cumul-0.75)))[0][0]
	return q1_time, q2_time, q3_time

