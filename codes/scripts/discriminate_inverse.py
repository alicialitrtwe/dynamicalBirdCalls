import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
from dynamicalbirdcalls.inverse import inverse
from dynamicalbirdcalls.config import processed_data_dir

os.chdir(processed_data_dir)
bird_name_list = np.load('segment_lists/bird_name_list.npy')
bird_name_list = np.repeat(bird_name_list, 3)

segment_len = 300
embed_dim_list = [10]
tau_list = [3]
pixels_list = [10]
spread_list = [0.01]
for embed_dim in embed_dim_list:
	for tau in tau_list:
		for pixels in pixels_list:
			for spread in spread_list:
				PI_list = np.load('TrainingData/PI_Len%s_Dim%s_Tau%s_Pix%s_Spd_%s.npy' %(segment_len, embed_dim, tau, pixels, spread))

period_list = np.load('TrainingData/period_list_%s.npy' %segment_len)
spectrum_list = np.load('TrainingData/spectrum_list_%s.npy' %segment_len)
fund_feat_list = np.load('TrainingData/fund_feat_list_%s.npy' %segment_len)

Ind = np.arange(len(bird_name_list))
np.random.seed(0) 
np.random.shuffle(Ind)

bird_name_list = bird_name_list[Ind]
PI_list = PI_list[Ind]
period_list = period_list[Ind]
spectrum_list = spectrum_list[Ind]
fund_feat_list = fund_feat_list[Ind]
form_list = fund_feat_list[:, :3]
fund_list = fund_feat_list[:, 3][:, None]
sal_list = fund_feat_list[:, 4][:, None]

no_nan_form = (np.sum(np.isnan(form_list), axis = 1) == 0) #357 nan
print(np.sum(no_nan_form))
no_nan_fund = (np.sum(np.isnan(fund_list), axis = 1) == 0) #124 nan
print(np.sum(no_nan_fund))
no_nan_idx = [d and m for d, m in zip(no_nan_fund, no_nan_form)]


feature_list = [period_list]
feature_name_list = ['PI']

for i in range(len(feature_list)):

	scores, p_values, n_high_score, n_counter = inverse(feature_list[i], bird_name_list, nonanIndFeature=no_nan_idx, cv=False, printijScores=False)
	#classi_perform = np.load('%s_classi_perfrom_Len_%s.npz' %(feature_name_list[i], segment_len))
	#scores = classi_perform['scores']
	#p_values = classi_perform['p_values']
	#n_high_score = classi_perform['n_high_score']
	#n_counter = classi_perform['n_counter']

