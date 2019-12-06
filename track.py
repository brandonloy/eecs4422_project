import os
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import time
from hog import hog
from spatialReliability import spatialMap
from hogFilter import hogFilter
from hogFilter import convFilt
from scipy import signal
from peakDetect import findPeak2d

def myThresh(img, val):
    peak = np.max(img)
    x = peak - (peak/100)*val
    res = np.where(img > x, np.ones(img.shape), np.zeros(img.shape))
    return res


vid = {1:'blurbody',
       2:'jump',
       3:'liquor',
       4:'skating2',
       5:'skating2',
       6:'surfer',
       7:'biker',
       8:'david',
       9:'MotorRolling',
       10:'bolt',
       11:'rubiks',
       12:'vite',
       13:'dudek',
       14:'tiger2',
       15:'cup',
       16:'Dancer2',
       17:'DragonBaby',
       18:'Soccer',
       19:'Trellis',
       20:'Coupon'
    }
box = {1:(400,48,87,319),
       2:(136,35,52,182),
       3:(256,152,73,210),
       4:(289,67,64,236),
       5:(347,58,103,251),
       6:(269, 129, 79, 158),
       7:(249, 93, 48, 120),
       8:(137, 44, 111, 116),
       9:(117,68,122,125),
       10:(336,165,26,61),
       11:(94, 276, 162, 137),
       12:(64, 108, 105, 185),
       13:(123,87,132,176),
       14:(32,60,68,78),
       15:(162, 224, 168, 140),
       16:(146, 36, 49, 189),
       17:(112, 65, 131, 267),
       18:(294, 122, 94, 104),
       19:(135, 40, 94, 125),
       20:(139, 59, 65, 102)
       }

#print(os.listdir(path))

seq = 15
selectroi = False
genPlot = False

#path = os.path.join(os.environ['HOME'],'Desktop','VOT','OTB100',vid[seq],vid[seq],'img')
path = os.path.join('OTB100',vid[seq],vid[seq],'img')
txtPath = os.path.join('my_out',vid[seq]+'.txt')
txt = open(txtPath,"w+")

cell = 4
count = 0
bbox = box[seq]
filt = []
feat = []
exp = []
resp = []
start = time.time()
for frame in os.listdir(path):
    framepath = os.path.join(path,frame)
    img = cv2.imread(framepath)
    try:
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except:
        break
    if count == 0:
    #first frame
        if selectroi:
            bbox = cv2.selectROI(img, False)
            cv2.destroyAllWindows()
        x,y,w,h = bbox
        print(bbox)
        print(((x+w//2)//cell,(y+h//2)//cell))
        #learn is the highest peak in the expected response
        #(not normalized)
        filt, feat, exp, learn = hogFilter(hsv, bbox, (cell,cell))
        resp = exp
    else:
    #find position in next frame
        resp = np.zeros(feat.shape)
        feat = hog(hsv,(cell,cell))

        # weights for channel reliability
        reliable = []
        maxx = []
        maxy = []
        #transform our box to hog space
        x,y,w,h = bbox
        x //= cell
        y //= cell
        w //= cell
        h //= cell
        cx = x + w//2
        cy = y + h//2
        #print(cx,cy)
        for i in range(0,9):
            #search new features with old filters
            resp[:,:,i], _ = convFilt(filt[:,:,i],feat[:,:,i])
            #find peaks in the response window of the bbox
            window = resp[y:y+h,x:x+w,i]
            xpeak, ypeak, peakVals = findPeak2d(window)

            
            #print(xpeak[0],ypeak[0])
##            print(xpeak)
##            print(ypeak)
##            plt.subplot(311)
##            plt.imshow(feat[:,:,i])
##            plt.subplot(312)
##            plt.imshow(window)
##            plt.subplot(313)
##            plt.imshow(resp[:,:,i])
##            plt.show()
            try:
                r_det = 1 - peakVals[1]/peakVals[0]
                r_learn = learn[i]/np.sum(learn)
                
                maxx.append(x + xpeak[0])
                maxy.append(y + ypeak[0])
                #divide by sum of learning reliabilities
                #reliable.append((r*learn[i])/np.sum(learn))
                reliable.append(r_learn)
                #print((x+xpeak[0],y+ypeak[0]),r_learn,r_det)
            except:
                maxx.append(0)
                maxy.append(0)
                reliable.append(0)

            #print((xpeak[0], ypeak[0], r))

##        for i in range(0,9):
##            plt.subplot(3,9,1 + i)
##            plt.imshow(feat[:,:,i])
##            plt.subplot(3,9,10 + i)
##            plt.imshow(resp[:,:,i])
##            plt.subplot(3,9,19 + i)
##            plt.imshow(exp[:,:,i])
##        plt.show()
        #----------------------------------------- Generate response plots
        if genPlot:
            foo = []
            foobar= []
            fu = []
            for i in range(0,9):
                foo.append(resp[:,:,i])
                foobar.append(filt[:,:,i])
                fu.append(exp[:,:,i])
            resps = np.hstack(tuple(foo))
            filts = np.hstack(tuple(foobar))
            exps = np.hstack(tuple(fu))
            comb = np.vstack((resps,exps))
            cv2.imshow("responses", comb)
            cv2.imshow("filter",filts)
        #calculate new object center
        newcx = 0
        newcy = 0
        #print(np.sum(reliable))
        #print(reliable)
        for i in range(0,9):
            #maximum peaks*channel reliability
            newcx += maxx[i]*reliable[i]
            newcy += maxy[i]*reliable[i]

        if (np.sum(reliable)>0):
            newcx /= np.sum(reliable)
            newcy /= np.sum(reliable)
        else:
            newcx = cx
            newcy = cy
        #print((cx,cy),(newcx, newcy))
        #calculate new bbox
        x = int(newcx - w/2)*cell
        y = int(newcy - h/2)*cell
        w *= cell
        h *= cell
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        bbox = (x,y,w,h)
        try:
            filt, feat, exp, learn = hogFilter(hsv, bbox, (cell,cell))
        except:
            print("BOX IS OUT OF BOUNDS, new box: " + str(bbox))
    #cv2.imshow('frame', img)
    imcopy = np.array(img)
    p1 = (x,y)
    p2 = (x+w,y+h)
    cv2.rectangle(imcopy, p1, p2, (255,0,0), 2, 1)
    patch = img[y:y+h,x:x+w]
    #cv2.imshow('roi', patch)
    cv2.imshow('track',imcopy)
    cv2.waitKey(1)
    txt.write(str(bbox)+'\n')
    count += 1
    
##    for i in range(0,9):
##        plt.subplot(3,9,1 + i)
##        plt.imshow(feat[:,:,i])
##        plt.subplot(3,9,10 + i)
##        plt.imshow(filt[:,:,i])
##        plt.subplot(3,9,19 + i)
##        plt.imshow(exp[:,:,i])
##    plt.show()
cv2.destroyAllWindows()
end = time.time()
print("elapsed: "+str(end-start)+'\tframes:'+str(count))
elapsed = end-start
txt.write('elapsed: ' + str(elapsed) + '\tframes: '+ str(count))
txt.close()
