import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import math as m
from hog import hog
from hogFilter import hogFilter
from spatialReliability import spatialMap
from scipy import signal

def findPeak2d(resp):
    """
    Finds local maxima in a 2d array

    Input:
    resp - result of convulting a filter and a channel feature

    Output:
    In each output array, index corresponds to the same point
    Arrays are sorted by peak magnitudes
    
    xpeak - x coordinates of peaks
    ypeak - y coordiantes of peaks
    peakVals - magnitude of peaks
    """
    xpeak = []
    ypeak = []
    peakVals = []
    thresh = 0.5
    #peaks = np.where(resp > thresh, np.ones(resp.shape), np.zeros(resp.shape))
    h, w = resp.shape
    peaks = np.zeros((h,w))
    #print(h,w)
    ksize = 3
    for i in range(ksize,h-ksize):
        for j in range(ksize,w-ksize):
            if (resp[i,j] == np.max(resp[i-ksize:i+ksize,j-ksize:j+ksize])) and (resp[i,j] > thresh):
                peaks[i,j] = 1
                xpeak.append(j)
                ypeak.append(i)
                peakVals.append(resp[i,j])

    peakVals = np.array(peakVals)
    xpeak = np.array(xpeak)
    ypeak = np.array(ypeak)
    inds = peakVals.argsort()
    xpeak = xpeak[inds[::-1]]
    ypeak = ypeak[inds[::-1]]
    peakVals = peakVals[inds[::-1]]


    return xpeak,ypeak,peakVals

