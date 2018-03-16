import numpy as np 
import cv2 as cv 


def find_corner(reg):
    """ input 4 corner 
        output in order top-left top-right bottom-right bottom-left     
    """
    reg = np.reshape(reg, (4,2))
    _res = np.zeros((4,2), dtype= "float32")

    # top-left bottom-right
    s = reg.sum(axis=1)
    _res[0] = reg[np.argmin(s)]
    _res[2] = reg[np.argmax(s)]

    # top-right bottom-left
    diff = np.diff(reg, axis=1)
    _res[1] = reg[np.argmin(diff)]
    _res[3] = reg[np.argmax(diff)]

    return _res


def check_wrong_pics(reg1, reg2):
    """ input 4 corner in order 
        output False if it cover another
    """

    x, y, u, v = reg1
    _x, _y, _u, _v = reg2

    if (x[0] <= _x[0] and x[1] <= _x[1] and u[0] >= _u[0] and u[1] >= _u[1]):
        return False
    
    return True


def warp_pics(img, reg):
    """ input img and 4 corner
        output warp img
    """
    (tl, tr, br, bl) = reg
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv.getPerspectiveTransform(reg, dst)
    warp = cv.warpPerspective(img, M, (maxWidth, maxHeight))

    return warp
        
