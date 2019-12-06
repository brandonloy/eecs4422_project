import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import math as m
from hog import hog
from hogFilter import hogFilter
from spatialReliability import spatialMap
from scipy import signal

#find peaks in the response
def findPeak2d1(resp):
    h,w = resp.shape
    xpeak = []
    ypeak = []
    for j in range(0,h):
        #loop through the rows
        x, _ = signal.find_peaks(resp[j,:], height = 0.9)
        for p in x:
            xpeak.append(p)
            ypeak.append(j)
    #a 2d peak will show itself more than once
    # so remove the duplicates
###---------------- Pop off duplicates
##    idx = 0   
##    while(idx < len(xpeak)-1):
##        if (abs(xpeak[idx] - xpeak[idx+1]) < 3) and (abs(ypeak[idx] - ypeak[idx+1]) < 3):
##            xpeak.pop(idx)
##            ypeak.pop(idx)
##        else:
##            idx += 1
    
#----------------Average Duplicates
    idx = 0
##    print(xpeak)
##    print(ypeak)
    while(idx < len(xpeak)-1):
        if (abs(xpeak[idx] - xpeak[idx+1]) <2) and (abs(ypeak[idx] - ypeak[idx+1] < 2)):
            # assign new value to idx, then pop idx+1
            xpeak[idx] = (xpeak[idx] + xpeak[idx+1])//2
            xpeak.pop(idx+1)
##            print(ypeak)
            ypeak[idx] = (ypeak[idx] + ypeak[idx+1])//2
            ypeak.pop(idx+1)
        else:
            idx += 1
##
##    print(xpeak)
##    print(ypeak)
            
    peakVals = []
    n = len(xpeak)
    for j in range(0,n):
        peakVals.append(resp[ypeak[j],xpeak[j]])
        
    #sort peaks based on magnitude
    peakVals = np.array(peakVals)
    xpeak = np.array(xpeak)
    ypeak = np.array(ypeak)
    inds = peakVals.argsort()
    xpeak = xpeak[inds[::-1]]
    ypeak = ypeak[inds[::-1]]
    peakVals = peakVals[inds[::-1]]
    print(xpeak)
    print(ypeak)

    return xpeak,ypeak,peakVals

def findPeak2d(resp):
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
##    print(xpeak)
##    print(ypeak)
##    plt.subplot(121)
##    plt.imshow(resp)
##    plt.subplot(122)
##    plt.imshow(peaks)
##    plt.show()

    return xpeak,ypeak,peakVals

##n = 2
##name = {1:'soccerGuy.jpg',
##        2:'jump.jpg',
##        3:'biker.jpg',
##        4:'surfer.jpg',
##        5:'girl.jpg',
##        6:'blurbody.jpg',
##        7:'skater.jpg'
##        }
##box = {1:(211, 144, 248, 261),
##       2:(136, 35, 52, 182),
##       3:(262, 94, 16, 26),
##       4:(275,137,23,26),
##       5:(57,21,31,45),
##       6:(400,48,87,319),
##       7:(57,21,31,45)
##       }
##
##
##img = cv2.imread(name[n])
##hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
###bbox = cv2.selectROI(img, False)
##bbox = (127, 65, 60, 95)
##print(bbox)
##x,y,w,h = bbox
##cx = (x + w//2)//4
##cy = (y + h//2)//4
##print((cx,cy))
##filt, feat, exp, learn = hogFilter(hsv, bbox, cell_size=(4,4))
###print(learn)
##
##for i in range(0,9):
##    xpeak, ypeak, peakvals = findPeak2d(exp[:,:,i])
##
####    plt.imshow(exp[:,:,0])
####    plt.show()
####for i in range(0,9):
####    plt.subplot(3,9,1 + i)
####    plt.imshow(filt[:,:,i])
####    plt.subplot(3,9,10 + i)
####    plt.imshow(feat[:,:,i])
####    plt.subplot(3,9,19 + i)
####    plt.imshow(exp[:,:,i])
####plt.show()
