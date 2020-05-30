import numpy as np

import os
try:
    user_paths = os.environ['PYTHONPATH']#.split(os.pathsep)
except KeyError:
    user_paths = []


import sys
print('\n'.join(sys.path))

import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot

from matplotlib.ticker import MultipleLocator, FixedLocator
import seaborn as sns

from dynamicalbirdcalls.config import processed_data_dir
from dynamicalbirdcalls.set_plot_params import plot_params