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


#feature_list = [PI_list, period_list, spectrum_list, fund_feat_list, form_list, fund_list, sal_list]
feature_name_list = ['PI', 'period', 'spectrum', 'fund_feat', 'form', 'fund', 'sal']

scores = []
for i in range(len(feature_name_list)):
	ClassiPerform = np.load('%s_classi_perfrom_Len_300.npz' %feature_name_list[i])
	scores.append(ClassiPerform['scores'])
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
plt.scatter(scores[0], scores[1])
plt.xlim([40,100])
plt.ylim([40,100])
plt.plot([40, 100], [40, 100])
plt.show()
