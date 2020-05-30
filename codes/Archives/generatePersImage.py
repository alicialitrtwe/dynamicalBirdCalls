from AttractorReconstructUtilities import TimeDelayReconstruct
from persim import PersImage
from ripser import Rips
import numpy as np
import matplotlib.pyplot as plt

def generatePD(allTimeDelayedSeg, segmentLength, embDim, tau, PCA_n_components, norm, savePD):
    rips = Rips(maxdim=1, coeff=2)
    diagrams_h1 = [rips.fit_transform(data)[1] for data in allTimeDelayedSeg]
    if savePD == True:
        np.save('PD_Len%s_Dim%s_Tau%s_PCA%s_%s.npy' %(segmentLength, embDim, tau, PCA_n_components, norm), diagrams_h1)
    return diagrams_h1

def generatePI(allPDs, segmentLength, embDim, tau, PCA_n_components, pixels, spread, norm, savePI):

    pim = PersImage(pixels=[pixels, pixels], spread=spread)
    imgs = pim.transform(allPDs)
    imgs_array = np.array([img.flatten() for img in imgs])
    if savePI == True:
        np.save('PI_Len%s_Dim%s_Tau%s_PCA%s_p%s_s%s_%s.npy' %(segmentLength, embDim, tau, PCA_n_components, pixels, spread, norm), imgs_array)
    return imgs_array