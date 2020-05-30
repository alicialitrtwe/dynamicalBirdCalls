from matplotlib import rc, rcParams
import matplotlib.pyplot as plt

# Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
#rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':10})

def plot_params():
  params = {
   'axes.labelsize': 12,
   'font.size': 12,
   'font.family': 'serif',
   'legend.fontsize': 12,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'mathtext.fontset': 'cm',
   'mathtext.rm': 'serif',
   'text.usetex': False,
   'figure.figsize': [4, 4]
   }
  rcParams.update(params)
  
def stylize_axes(ax, title=None, xlabel=None, ylabel=None, xticks=[], yticks=[], xticklabels=[], yticklabels=[]):
    """Customize axes spines, title, labels, ticks, and ticklabels."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_tick_params(top='off', direction='out', width=1)
    ax.yaxis.set_tick_params(right='off', direction='out', width=1)
    
    ax.set_title(title)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)