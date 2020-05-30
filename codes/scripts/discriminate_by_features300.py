import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator
from dynamicalbirdcalls.pairwise_logi_regression import pairwise_logi_regression
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


feature_list = [PI_list, period_list, spectrum_list, fund_feat_list, form_list, fund_list, sal_list]
feature_name_list = ['PI', 'period', 'spectrum', 'fund_feat', 'form', 'fund', 'sal']

fig = plt.figure(figsize=(len(feature_list)*2, 5))
ax = fig.add_subplot(111)
ax.set_xlim(0, len(feature_list)+.4)
ax.set_ylim(30, 102)
plt.axhline(y=50, color = 'k', linewidth=2, alpha=0.5 )
colors = sns.color_palette("Set1", n_colors=len(feature_list), desat=0.76)

for i in range(len(feature_list)):

	scores, p_values, n_high_score, n_counter = pairwise_logi_regression(feature_list[i], bird_name_list, nonanIndFeature=no_nan_idx, cv=True, printijScores=False)
	np.savez('%s_classi_perfrom_Len_%s.npz' %(feature_name_list[i], segment_len), scores=scores, p_values=p_values, n_high_score=n_high_score, n_counter=n_counter)
	#classi_perform = np.load('%s_classi_perfrom_Len_%s.npz' %(feature_name_list[i], segment_len))
	#scores = classi_perform['scores']
	#p_values = classi_perform['p_values']
	#n_high_score = classi_perform['n_high_score']
	#n_counter = classi_perform['n_counter']

	sig_idx = np.where(p_values<0.05)[0]
	no_sig_idx = np.where(p_values>=0.05)[0]
	x_random = np.random.random(len(scores))*0.4 + i + 0.6
	ax.scatter(x_random[sig_idx], scores[sig_idx], s = 10, c=colors[i], alpha=0.8)
	ax.scatter(x_random[no_sig_idx], scores[no_sig_idx], s = 10, facecolors='none', edgecolors=colors[i], alpha=0.8)
	ax.scatter(x_random, scores, s = 10, c=colors[i], alpha=0.8)
	ax.scatter(x_random, scores, s = 10, facecolors='none', edgecolors=colors[i], alpha=0.8)
	
#ax.xaxis.set_tick_params(top='off', direction='in', width=0.4)
#ax.yaxis.set_tick_params(right='off', direction='in', width=0.4)
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#
#ax.set_xlabel('Feature Space')
#ax.set_ylabel('Percent Accuracy')
#
#ax.xaxis.set_major_locator(FixedLocator([0.8, 1.8, 2.8, 3.8, 4.8, 5,8, 6,8]))
#ax.yaxis.set_major_locator(FixedLocator([30, 50, 100]))
#
ax.set_xticklabels(feature_name_list)
plt.show()

#fig.savefig('Fig7 classi_perform.svg', dpi=300, bbox_inches='tight', transparent=True)