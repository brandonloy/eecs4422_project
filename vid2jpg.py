import numpy as np
import cv2
import os

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def vid2jpg(vidPath, shrink = 1):
    """
    Input:
    vidPath - path to video file
    shrink - factor to shrink video by
    """
    cap = cv2.VideoCapture(vidPath)
    count = 1
    videoName = os.path.basename(os.path.normpath(vidPath))
    videoName = videoName.split(".")
    name = videoName[0]
    outpath = os.path.join(os.getcwd(),name)
    try:
        os.mkdir(outpath)
    except:
        pass
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not(ret):
            break
        frame = cv2.resize(frame,None,fx=1/shrink,fy=1/shrink)
        frame = rotate_image(frame, 270)
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        #cv2.imshow('frame',frame)
        path = os.path.join(os.getcwd(),name,"{0:0=4d}".format(count)+'.jpg')
        cv2.imwrite(path,frame)
        count += 1
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print(str(count-1) + ' frames written to \"' + name + '\" directory')


#vid2jpg('C:\\Users\\brand\\Google Drive\\ASchool\\4422\\project\\cup.MOV', shrink = 3)
