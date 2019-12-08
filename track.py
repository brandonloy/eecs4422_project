import os
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import time
from hog import hog
from spatialReliability import spatialMap
from features import convFilt
from features import getFeatures
from scipy import signal
from peakDetect import findPeak2d




def csrTrack(viddir, genPlot = False):
    """
    Input:
    viddir - directory with video frames as jpg files.
             Jpg's must be named so that its alphanumerical order
             matches the chronological order of the frames
             
    genPlot - Set to True to display channel activation maps
    """
    cell = 4
    count = 0
    filt = []
    feat = []
    exp = []
    resp = []
    start = time.time()

    path = os.path.join(viddir)

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
            bbox = cv2.selectROI(img, False)
            cv2.destroyAllWindows()
            x,y,w,h = bbox
            print('Initial Box: ' + str(bbox))
            print('Initial Center: ' + str(((x+w//2)//cell,(y+h//2)//cell)))
            #learn is the highest peak in the expected response
            #(not normalized)
            filt, feat, exp, learn = getFeatures(hsv, bbox, (cell,cell))
            resp = exp
        else:
        #find position in next frame
            resp = np.zeros(feat.shape)
            _, feat, _, _ = getFeatures(hsv, bbox,(cell,cell))

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
            for i in range(0,10):
                #search new features with old filters
                resp[:,:,i], _ = convFilt(filt[:,:,i],feat[:,:,i])
                #find peaks in the response window of the bbox
                window = resp[y:y+h,x:x+w,i]
                xpeak, ypeak, peakVals = findPeak2d(window)  

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

            #----------------------------------------- Generate response plots
            if genPlot:
                foo = []
                foobar= []
                fu = []
                for i in range(0,10):
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
            for i in range(0,10):
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
                filt, feat, exp, learn = getFeatures(hsv, bbox, (cell,cell))
            except:
                print("BOX IS OUT OF BOUNDS, new box: " + str(bbox))
        imcopy = np.array(img)
        p1 = (x,y)
        p2 = (x+w,y+h)
        cv2.rectangle(imcopy, p1, p2, (255,0,0), 2, 1)
        patch = img[y:y+h,x:x+w]
        cv2.imshow('track',imcopy)
        cv2.waitKey(1)
        count += 1
        

    cv2.destroyAllWindows()
    end = time.time()
    print("elapsed: "+str(end-start)+'\tframes:'+str(count))
    elapsed = end-start

