import seaborn as sns
import matplotlib.pyplot as plt
plt.close('all')
sns.palplot(sns.color_palette("Set1", n_colors=3, desat=0.6))
#print(sns.palplot(sns.color_palette("Set1", n_colors=8, desat=.5)))
plt.show()