import cv2 as cv
import numpy as np


def watershed():
    img = cv.imread('.\\data\\test\\fruits1.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255,
                               cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(),
                                255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    return img


def selective_search(im, type='f'):
    # Multithread the search for better performance
    cv.setUseOptimized(True)
    cv.setNumThreads(4)

    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)

    # Fast speed
    if type == 'f':
        ss.switchToSelectiveSearchFast()

    # High quality
    if type == 'q':
        ss.switchToSelectiveSearchQuality()

    print("Searching...")
    # Run selective search
    rects = ss.process()

    # Return the regions as a list of rectangle co-ordinates
    return rects
