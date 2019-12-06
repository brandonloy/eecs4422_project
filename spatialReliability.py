import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import time
import os 



## is a modified Epanechnikov kernel,
##k(r; sigma) = 1 − (r/sigma)2, with size parameter sigma equal to the
##minor bounding box axis and clipped to interval [0.5, 0.9]
##such that the object prior probability at center is 0.9 and
##changes to a uniform prior away from the center (Figure 2).
priorLim = 0.5
def epanK(n):
    #assuming n is odd
    sigma = (n - 1)/ 2
    x = np.arange(-sigma, sigma+1,1)
    k = 1 - (x/sigma)**2

    res = np.outer(k, k)
    ones = np.ones(res.shape)
    res = np.where(res < priorLim, ones*priorLim, res)
    res = np.where(res > 0.9, ones*0.9, res)

    return res
    
def spatialMap1(img, bbox):
    """
    Creates a mask for filter creation
    In the original CSR-DCF, this mask is used as a
    prior distribution to compute the MAP estimate of the
    classification of background foreground pixels
    
    Input:
    img - HSV image
    bbox - Region that contains the object

    Output:
    out - Binary Mask
    """
    # select object region from image
    #bbox = cv2.selectROI(img, False)
    x, y, w, h = bbox
    #x,y,w,h = (240, 69, 98, 119)
    if (w % 2 != 1):
        w = w - 1
    if (h % 2 != 1):
        h = h - 1
    patch = img[y:y+h,x:x+w]

    priorImg = np.ones(img[:,:,0].shape)*priorLim
    prior = np.ones(patch[:,:,0].shape)*priorLim
     
    if (w < h):
        n = w
        epan = epanK(n)
        foo = (h-n)//2
        # foo is the half of the pixles that arent covered by epan
        priorImg[y+foo:(y+h)-foo,x:(x+w)] = epan
        prior[foo:h-foo,:] = epan
    else:
        n = h
        epan = epanK(n)
        foo = (w-n)//2
        priorImg[y:y+h,x+foo:(x+w)-foo] = epan
        prior[:,foo:w-foo] = epan
    
    objHist = cv2.calcHist([patch], [0,1], None, [180, 256], [0, 180, 0, 256])
    imgHist = cv2.calcHist([img], [0,1], None, [180, 256], [0, 180, 0, 256])

    cv2.normalize(objHist,objHist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([img],[0,1],objHist,[0,180,0,256],1)
    #dst is the appearance likelihood
    # Now convolute with circular disc (dilate)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(dst,-1,disc,dst)
    #dst = cv2.dilate(dst, disc, iterations = 1)
    # calculate ratio of bg/fg size
    fg = h*w
    hi, wi, sl = img.shape
    bg = (hi*wi) - fg
    ratio = fg/bg

    res = (dst*ratio)
    res = np.multiply(res, priorImg)
    out = res
    out = cv2.normalize(res, out, 0, 1, cv2.NORM_MINMAX)
    
    out*=255
    out = out.astype(np.uint8)

    return out


def spatialMap(img, bbox):
    """
    Binary Mask used in filter creation.
    Segments background and foreground using a canny based approach

    Input:
    img - HSV image
    bbox - Region that contains the object

    Output:
    out - Binary Mask
    """
    x, y, w, h = bbox
    #x,y,w,h = (240, 69, 98, 119)
    if (w % 2 != 1):
        w = w - 1
    if (h % 2 != 1):
        h = h - 1
    patch = img[y:y+h,x:x+w]

    thresh1 = 200
    thresh2 = 30
    ksize = 3


    edge = cv2.Canny(img[y:y+h,x:x+w,2], threshold1 = thresh1, threshold2 = thresh2, apertureSize = ksize)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE,kernel,iterations=4)
    edge = cv2.dilate(edge, kernel,iterations=1)

    Cancon = cv2.findContours(edge.astype('uint8'), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]
    #contour_img = np.zeros(canny_img.shape)
    mask = np.zeros(edge.shape)
    for contour in Cancon:
        cv2.drawContours(mask, [contour], -1, (255,255,255), cv2.FILLED)


    out = np.zeros(img[:,:,0].shape)
    out[y:y+h,x:x+w] = mask
    
    cv2.imshow('mask',out)
    return out



