import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import math as m
from hog import hog
from spatialReliability import spatialMap
from scipy import signal


def hogFilter(hsv, bbox, cell_size=(8,8)):
    x,y,w,h = bbox
    res = spatialMap(hsv, bbox)
    mask = np.zeros(hsv.shape,np.uint8)
    mask[:,:,0] = res
    mask[:,:,1] = res
    mask[:,:,2] = res
##    print(res.dtype)
##    print(hsv.dtype)
##    print(mask.dtype)
    maskedImg = cv2.bitwise_and(hsv, mask)
##    plt.imshow(mask)
##    plt.show()
    patch = maskedImg[y:y+h,x:x+w]

    #cv2.imshow('',res)
    #plt.imshow(res)
    #plt.show()

# Filter should be the whole HOG image, but i started off with a patch
    filt = hog(patch, cell_size)
    #filt = hog(maskedImg, cell_size)

    feat = hog(hsv, cell_size)

    maskedFilt = np.array(feat)

    feat_y, feat_x, _ = feat.shape
    
    res = cv2.resize(res,(feat_x, feat_y))
    #print(hsv.shape, feat.shape, res.shape)
    for i in range(0,9):
        maskedFilt[:,:,i] = np.multiply(feat[:,:,i],res)
    #filt = maskedFilt
    #print(filt.shape)
    
    #9 bins
    exp = np.zeros(feat.shape)
    learn = np.zeros(9)
    for i in range(0, 9):
##        h = (filt[:,:,i]-np.mean(filt[:,:,i]))/np.sum(filt[:,:,i])
##        exp[:,:,i] = cv2.filter2D(feat[:,:,i],-1,h)
        exp[:,:,i], learn[i] = convFilt(filt[:,:,i],feat[:,:,i])
##        plt.subplot(3,3,1+i)
##        plt.imshow(filt[:,:,i])
##    plt.show()
    #print(learn)
    learn = cv2.normalize(learn, learn, 0, 1, cv2.NORM_MINMAX)
    
    return filt, feat, exp, learn

def convFilt(filt, feat):
    """
    Inputs:
    filt - single channel, unnormalized
    feat - feature image
    
    Returns:
    exp - expected filter response
    peak - maximum response value before normalization
    """
##    h = (filt-np.mean(filt))/np.sum(filt)
##    exp = cv2.filter2D(feat,-1,h)

    filt = filt[::-1,::-1]
    
    h = (filt-np.mean(filt))/max(np.sum(filt),1)

    exp = signal.fftconvolve(feat, h, mode='same')
    out = exp
    peak = np.max(exp)
    out = cv2.normalize(exp, out, 0, 1, cv2.NORM_MINMAX)
    
    return out, peak


#C:\Users\brand\Desktop\VOT\OTB100\Jump\Jump\img


##
##n = 7
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
##
##
##
##
##bbox = cv2.selectROI(img, False)
###bbox = box[n]
##print(bbox)
###x,y,w,h = bbox
##h, w, _ = hsv.shape
##filt, feat, exp, learn = hogFilter(hsv, bbox, cell_size=(4,4))
##
##feature = feat[:,:,5]
##filtr = filt[:,:,5]
##response = exp[:,:,5]
##feature = cv2.normalize(feat[:,:,5],feature,0,255,cv2.NORM_MINMAX)
##filtr = cv2.normalize(filt[:,:,5],filtr,0,255,cv2.NORM_MINMAX)
##response = cv2.normalize(exp[:,:,5],response,0,255,cv2.NORM_MINMAX)
####cv2.imwrite('feature.jpg',feature)
####cv2.imwrite('filter.jpg',filtr)
####cv2.imwrite('expectedResponse.jpg',response)
####
##
##
##
####img2 = cv2.imread('skater2.jpg')
####hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
####h, w, _ = hsv2.shape
#####(0,0,h,w)
####filt2, feat2, exp2, _ = hogFilter(hsv2, bbox, cell_size=(4,4))
####
####
####resp = np.zeros(feat2.shape)
####
####
####for i in range(0, 9):
####    resp[:,:,i],_ = (convFilt(filt[:,:,i], feat2[:,:,i]))
####
####
####
####response = cv2.normalize(exp[:,:,5],resp,0,255,cv2.NORM_MINMAX)
##print(learn)
##
##for i in range(0,9):
##    plt.subplot(3,9,1 + i)
##    plt.imshow(filt[:,:,i])
##    plt.subplot(3,9,10 + i)
##    plt.imshow(exp[:,:,i])
##    plt.subplot(3,9,19 + i)
##    plt.imshow(exp[:,:,i])
##plt.show()
