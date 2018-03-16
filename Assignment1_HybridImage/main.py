import cv2 as cv
import numpy as np 
from mymethod import *

img = cv.imread('image_assignment.jpg')
orig = cv.imread('image_assignment.jpg')

#resize img
ratio = 0.3
height, width = img.shape[:2]
img = cv.resize(img, None, fx=ratio, fy=ratio , interpolation= cv.INTER_AREA)


#____localize border

RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
RGB = cv.medianBlur(RGB, 5)
lower_color = np.array([180,186,110])
upper_color = np.array([230,230,230])

mask = cv.inRange(RGB, lower_color, upper_color)
res = cv.bitwise_and(RGB, RGB, mask= mask)

_, cnts, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv.contourArea ,reverse=True)[:5]

#____find img

pics = []
for cnt in cnts:
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.03 * peri, True)

    if (len(approx) == 4 ):
        pics.append(approx)

#____cropimg
# top-left top-right bottom-right bottom-left

new_pics = []
for c in pics:
    new_pics.append(find_corner(c))

_pics = []
for i in range(len(new_pics)):
    check = True
    for j in range(i+1, len(new_pics)):
        if (not check_wrong_pics(new_pics[i], new_pics[j])):
            check = False
            break
    if (check):
        _pics.append(new_pics[i] )
    
warp = []
for c in _pics:
    warp.append(warp_pics(img.copy(), c))
    # cv.imshow('warp', warp[-1])
    # cv.waitKey(0)

cv.imwrite('pic1.jpg', warp[1])
cv.imwrite('pic2.jpg', warp[0])


# for c in _pics:
#     cv.drawContours(img, [c], -1, (0,225,0), 3)

#____

# cv.imshow('frame' , img)
# cv.imshow('warp', warp)
# cv.waitKey(0)