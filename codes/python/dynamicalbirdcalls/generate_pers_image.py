import numpy as np
import matplotlib.pyplot as plt

from persim import PersImage
from ripser import Rips


def generate_PD(reconstru_list, emb_dim, tau):

    rips = Rips(maxdim=1, coeff=2)
    PD_list = [rips.fit_transform(data)[1] for data in reconstru_list]

    return np.array(PD_list)

def generate_PI(PD_list, emb_dim, tau, pixels, spread):

    pim = PersImage(pixels=[pixels, pixels], spread=spread)
    PI_list = pim.transform(PD_list)
    PI_list = np.array([img.flatten() for img in PI_list])
    
    return np.array(PI_list)