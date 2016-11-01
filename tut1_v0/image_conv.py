# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:28:45 2016

@author: adnen
"""

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as pl
def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)
# Set up grid and test data
data =numpy.matrix([[1, 1, 1,1],[0, 0, -0, -0],[-0, -0, -0, -0],[-1, -1,-1,-1]])
plt.figure(figsize=(4, 4))
nice_imshow(pl.gca(), data, cmap=cm.binary_r)

import scipy.misc
lena = scipy.misc.lena()

eye = lena[250:280,250:280]

plt.figure(figsize=(20, 20))
nice_imshow(pl.gca(), lena, cmap=cm.binary_r)

from scipy import signal
corr = signal.correlate2d(lena, data, boundary='symm', mode='same')
eye = corr[250:280,250:280]
corr2 = signal.correlate2d(corr, eye, boundary='symm', mode ='same')
plt.figure(figsize=(20, 20))
nice_imshow(pl.gca(), corr2, cmap=cm.binary_r)