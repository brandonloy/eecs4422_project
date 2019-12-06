import numpy as np
import cv2
import matplotlib.pyplot as plt
# Adapted from
# https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python
def hog(img, cell_size = (8,8), block_size = (2, 2), nbins = 9):
    # winSize is the size of the image cropped to an multiple of the cell size
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
    hog_feats = hog.compute(img)\
                   .reshape(n_cells[1] - block_size[1] + 1,
                            n_cells[0] - block_size[0] + 1,
                            block_size[0], block_size[1], nbins) \
                   .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
    # hog_feats now contains the gradient amplitudes for each direction,
    # for each cell of its group for each group. Indexing is by rows then columns.

    gradients = np.zeros((n_cells[0], n_cells[1], nbins))

    # count cells (border cells appear less often across overlapping groups)
    cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

    for off_y in range(block_size[0]):
        for off_x in range(block_size[1]):
            gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                      off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                hog_feats[:, :, off_y, off_x, :]
            cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                       off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

    # Average gradients
    gradients /= cell_count

    return gradients





 # number of orientation bins
#for every 8x8 cell of the image, generate 64 gradient vectors
#(1 for every pixel) represent these gradients in a histogram
# of 9 bins
#
# For every 2x2 block of cells, normalize using stride 1
# this way the nomralization takes into account neighboring blocks

#output is 9 bins for every cell


##img = cv2.imread('skater.jpg')
##
###img = cv2.cvtColor(cv2.imread("skater.jpg"), cv2.COLOR_BGR2GRAY)
##cell_size = (8, 8)  # h x w in pixels
##block_size = (2, 2)  # h x w in cells
##nbins = 9 
##
##bbox = cv2.selectROI(img, False)
##x, y, w, h = bbox
##if (w % 2 != 1):
##    w = w - 1
##if (h % 2 != 1):
##    h = h - 1
##patch = img[y:y+h,x:x+w]
##gradients = hog(img)
##print(gradients.shape)
##print(patch.shape)
##
##
##
##for i in range(0,9):
##    plt.subplot(331 + i)
##    plt.imshow(gradients[:,:,i])
####    plt.pcolor(gradients[:, :, i])
####    plt.gca().invert_yaxis()
####    plt.gca().set_aspect('equal', adjustable='box')
####    plt.colorbar()
##plt.show()
##

