import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import time
import os 


vid = {1:'blurbody',
       2:'jump',
       3:'liquor',
       4:'skating2',
       5:'skating2',
       6:'surfer',
       7:'biker',
       8:'david',
       9:'MotorRolling',
       10:'bolt'
    }
box = {1:(400,48,87,319),
       2:(136,35,52,182),
       3:(256,152,73,210),
       4:(289,67,64,236),
       5:(347,58,103,251),
       6:(275,137,23,26),
       7:(262,94,16,26),
       8:(129,80,64,78),
       9:(117,68,122,125),
       10:(336,165,26,61)
       }


# This code allows you to specify an object in the foreground
# The image will be segmented based on color histograms
# https://docs.opencv.org/3.4/dc/df6/tutorial_py_histogram_backprojection.html


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
##    print(res)
##    plt.imshow(res, cmap='gray')
##    plt.show()
    return res
    
def spatialMap1(img, bbox):
    """
    img: HSV image
    Outputs a binary mask used in filter creation
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

    #https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/back_projection/back_projection.html
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

    dot = priorImg*255
    dot = np.where(dot > 130, np.ones(dot.shape)*255,np.zeros(dot.shape))
    dot = dot.astype(np.uint8)
##    print(priorImg.dtype)
##    plt.imshow(out)
##    plt.show()

    #out = np.where(out > priorImg, np.ones(out.shape)*0.5, np.zeros(out.shape))
    #   res is equation 3 in the paper
    #cv2.imshow("mask_spatrel",out)
    #cv2.imshow("bakproj",dst)
    return out




def spatialMap1(img, bbox):
    """
    GRABCUT 
    img: HSV image
    Outputs a binary mask used in filter creation
    """
    k = 4
    smol = cv2.resize(img,None,fx=1/k,fy=1/k)
    x,y,w,h = bbox
    x//=k
    y//=k
    w//=k
    h//=k
    rect = (x,y,w,h)
    mask = np.zeros(smol.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(smol,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    out = cv2.resize(mask2,None,fx=k,fy=k)
    out*=255
    return out

def spatialMap(img, bbox):
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
    
##    foo = 1
##    bar = 3
##    plt.subplot(foo,bar, 1)
##    plt.imshow(cv2.cvtColor(patch,cv2.COLOR_HSV2RGB))#(out)
##    plt.subplot(foo, bar, 2)
##    plt.imshow(edge)
##    plt.subplot(foo, bar, 3)
##    plt.imshow(out)
##    plt.show()
    cv2.imshow('mask',out)
    return out

##seq = 4
##path = os.path.join('OTB100',vid[seq],vid[seq],'img','0001.jpg')
##print(path)
##img = cv2.imread(path)
##poop = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
##
###bbox = cv2.selectROI(img, False)
##bbox = box[seq]
###print(bbox)
##start = time.time()
##res = spatialMap(poop, bbox)
##end = time.time()
##
##print(end-start)
##print(res.shape)
##print(img.shape)
##
##plt.imshow(res)
##plt.show()
##

