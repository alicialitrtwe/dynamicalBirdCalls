import numpy as np
import os
import seaborn as sns
from pwLogiRegression import pwLogiRegression
from findPeriods import findPeriod
from findSpectrum import findSpectrum
import matplotlib.pyplot as plt
import seaborn as sns
from generatePersImage import generatePD, generatePI

from matplotlib.ticker import MultipleLocator, FixedLocator
from sciPlot import updateParams
from confidenceInterval import mean_confidence_interval
updateParams()
# 10, 12 spectral best 2, 14 second best 10 22 third best 6 18 fourth best       
# 5m, 18 topology best; 5m, 21 second best  17m 21    (13m 18)   (17m 18) (14 20m) (5m 14) (5m, 16)

os.chdir('/Users/Alicia/Desktop/BirdCallProject/Freq_250_12000/dataAnalysis')

allFeaturesNameL = ['PI', 'Spectral', 'Fund', 'Formant',  "Period", 'PeriodPI']

scores = []
for i in range(len(allFeaturesNameL)):
	ClassiPerform = np.load('%sClassiPerfrom.npz' %allFeaturesNameL[i])
	scores.append(ClassiPerform['scores'])
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
plt.scatter(scores[2], scores[3])
plt.xlim([40,100])
plt.ylim([40,100])
plt.plot([40, 100], [40, 100])
plt.show()
