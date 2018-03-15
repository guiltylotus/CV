import cv2 as cv
import numpy as np 


img = cv.imread('image_assignment.jpg')
orig = img.copy

#resize img
height, width = img.shape[:2]
img = cv.resize(img, None, fx=0.3, fy=0.3 , interpolation= cv.INTER_AREA)


#____localize border

RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
RGB = cv.medianBlur(RGB, 5)
lower_color = np.array([180,186,110])
upper_color = np.array([230,230,230])

mask = cv.inRange(RGB, lower_color, upper_color)
res = cv.bitwise_and(RGB, RGB, mask= mask)

_, cnts, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv.contourArea ,reverse=True)[:5]

# #____find img

pics = []
for cnt in cnts:
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.03 * peri, True)

    print(len(approx))
    if (len(approx) >= 4 and len(approx) <= 4):
        pics.append(approx)
        print(approx)

new_img = img.copy()
for c in pics:
    cv.drawContours(new_img, [c], -1, (0,225,0), 3)

cv.imshow('frame' , new_img)
cv.imshow('mask', mask)
cv.waitKey(0)