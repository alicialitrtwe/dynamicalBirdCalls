import matplotlib.pyplot as plt
import matplotlib
colorName = []
import os 
import numpy as np
colorName.append('#dd7777')
os.chdir('/Users/Alicia/AnacondaProjects/CurrentProject/PlotPaper')
cmap=matplotlib.cm.get_cmap('viridis', 20)
for i in range(cmap.N):
    rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
    colorName.append(matplotlib.colors.rgb2hex(rgb))

plt.figure()
for i in range(len(colorName)):
	plt.axvline(x=i, color = colorName[i], linewidth=3)
plt.show()
np.save('colorName.npy', colorName)