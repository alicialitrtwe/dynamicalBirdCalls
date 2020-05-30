import numpy as np
import os
from pwLogiRegression import pwLogiRegression
from findPeriods import findPeriod
from findSpectrum import findSpectrum
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FixedLocator
from sciPlot import updateParams
from confidenceInterval import mean_confidence_interval
updateParams()

os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis')

allSegments = np.load('allSegment_Len500.npy')
birdName = np.load('birdName.npy')
birdName = np.repeat(birdName, 3)

SegmentLength = 500
norm = 'Normed'
#allPeriods = findPeriod(allSegments, SegmentLength, norm, avgPeriodLen=65)
#allSpectrums = np.array([findSpectrum(Seg, 44100, f_high=22040) for Seg in allSegments])

norm = 'Normed2Max'
segmentLengthL = [500]
embDimL = [5, 10, 25]
tauL = [3]
PCA_n_components = 0 
pixels = 15
spread = 1
allPIs = []
for Len in range(len(segmentLengthL)):
	for Dim in range(len(embDimL)):
		for Tau in range(len(tauL)):
			segmentLength = segmentLengthL[Len]
			embDim = embDimL[Dim]
			tau = tauL[Tau]
			if PCA_n_components < embDim and embDim*tau < 300:								
				allPIs.append(np.load('PI_Len%s_Dim%s_Tau%s_PCA%s_p%s_s%s_%s_Final.npy' %(segmentLength, embDim, tau, PCA_n_components, pixels, spread, norm)))

#allFunds = np.load('fund_Len500.npy')[:,None]
#allFormants = np.load('formant_Len500.npy')[:, :2]
#Ind = np.arange(len(birdName))
#np.random.seed(41) 
#np.random.shuffle(Ind)
#
#birdName = birdName[Ind]
#allFunds = allFunds[Ind]
#for i in range(len(allPIs)):
#	allPIs[i] = allPIs[i][Ind]
#allFormants = allFormants[Ind]
#allPeriods = allPeriods[Ind]
#allSpectrums = allSpectrums[Ind]
#
#nonanIndFund = (np.sum(np.isnan(allFunds), axis = 1) == 0)
#nonanIndForm = (np.sum(np.isnan(allFormants), axis = 1) == 0)
#nonanInd = [d and m for d, m in zip(nonanIndFund, nonanIndForm)]

#allFeaturesL = [allPIs[1], allSpectrums, allFormants, allFunds, allPeriods]
allFeaturesL = [allPIs[0], allPIs[1], allPIs[2]]
#allFeaturesNameL = ['PI', 'Specral', 'Fund', 'Formant',  "Period"]
#allFeaturesNameL = ['PI']
#allFeaturesL = [allPIs[1]]
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.set_xlim(0, 5.4)
ax.set_ylim(30, 102)
plt.axhline(y=50, color = 'k', linewidth=2, alpha=0.5 )
colorChosen = sns.color_palette("Set1", n_colors=5, desat=0.76)
for i in range(len(allFeaturesNameL)):
	scores, pValues, nhighScores, ncounter = pwLogiRegression(allFeaturesL[i], birdName, nonanIndFeature=nonanInd, cv=True, printijScores=False)
	#np.savez('%sClassiPerfrom.npz' %allFeaturesNameL[i], scores=scores, pValues=pValues, nhighScores=nhighScores, ncounter=ncounter)
	#ClassiPerform = np.load('%sClassiPerfrom.npz' %allFeaturesNameL[i])
	scores = ClassiPerform['scores']
	pValues = ClassiPerform['pValues']
	nhighScores = ClassiPerform['nhighScores']
	ncounter = ClassiPerform['ncounter']
	sigIdx = np.where(pValues<0.05)[0]
	noSigIdx = np.where(pValues>=0.05)[0]
	print(nhighScores, ncounter)
	xRandom = np.random.random(len(scores))*0.4 + i + 0.6
	ax.scatter(xRandom[sigIdx], scores[sigIdx], s = 10, c=colorChosen[i], alpha=0.8)
	ax.scatter(xRandom[noSigIdx], scores[noSigIdx], s = 10, facecolors='none', edgecolors=colorChosen[i], alpha=0.8)
	mean, h = mean_confidence_interval(scores)
	ax.errorbar(0.8+i, mean, yerr=h, ecolor='k', capsize=6, elinewidth=4, fmt='ok')
	ax.boxplot(scores, positions=range(i,i+1))
ax.xaxis.set_tick_params(top='off', direction='in', width=0.4)
ax.yaxis.set_tick_params(right='off', direction='in', width=0.4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Feature Space')
ax.set_ylabel('Percent Accuracy')

ax.xaxis.set_major_locator(FixedLocator([0.8, 1.8, 2.8, 3.8, 4.8]))
ax.yaxis.set_major_locator(FixedLocator([30, 50, 100]))

ax.set_xticklabels(['Topology', 'Spectral',  'Fund.', 'Formant', 'Period'])
plt.show()
#fig.savefig('Fig7 ClassiPerform.svg', dpi=300, bbox_inches='tight', transparent=True)