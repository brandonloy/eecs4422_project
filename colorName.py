import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import time
import os


def colorFeat(img, bbox):
    """
    Probability map of object likelihood based on color histogram

    Input:
    img - HSV image
    bbox - ROI of object

    Output:
    dst - Probability map of object location
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
    # only use a square in the center of the bbox
    # for the histogram creation
    # this is because the borders of the bbox contain
    # background pixels
    fgPatch = np.zeros(patch.shape)
    if (w < h):
        cy = h//2
        hw = w//2
        subpatch = patch[cy-hw:cy+hw,0:w]
    else:
        cx = w//2
        hh = h//2
        subpatch = patch[0:h,cx-hh:cx+hh]
##    plt.imshow(subpatch)
##    plt.show()
        
    
    objHist = cv2.calcHist([subpatch], [0,1], None, [180, 256], [0, 180, 0, 256])
    imgHist = cv2.calcHist([img], [0,1], None, [180, 256], [0, 180, 0, 256])

    #https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/back_projection/back_projection.html
    cv2.normalize(objHist,objHist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([img],[0,1],objHist,[0,180,0,256],1)
    #dst is the appearance likelihood
    # Now convolute with circular disc (dilate)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    cv2.filter2D(dst,-1,disc,dst)

    return dst



