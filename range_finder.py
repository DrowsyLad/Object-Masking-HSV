import cv2
import numpy as np
import json
import os.path
from os import path

camera = cv2.VideoCapture(0)

def nothing(x):
    pass

with open('red_hsv.json', 'r') as openfile:
 
    # Reading from json file
    red_hsv = json.load(openfile)

mask_hsv_path = 'mask_hsv.json'
if path.exists(mask_hsv_path):
    with open(mask_hsv_path, 'r') as openfile:
    
        # Reading from json file
        mask_hsv = json.load(openfile)
else:
    mask_hsv = red_hsv

with open('circle_params.json', 'r') as openfile:
 
    # Reading from json file
    circle_params = json.load(openfile)


cv2.namedWindow('hsv red')
cv2.createTrackbar('H Lower','hsv',red_hsv['H_Lower'],179,nothing)
cv2.createTrackbar('H Higher','hsv',red_hsv['H_Higher'],179,nothing)
cv2.createTrackbar('S Lower','hsv',red_hsv['S_Lower'],255,nothing)
cv2.createTrackbar('S Higher','hsv',red_hsv['S_Higher'],255,nothing)
cv2.createTrackbar('V Lower','hsv',red_hsv['V_Lower'],255,nothing)
cv2.createTrackbar('V Higher','hsv',red_hsv['V_Higher'],255,nothing)

cv2.namedWindow('hsv mask')
cv2.createTrackbar('H Lower','mask',mask_hsv['H_Lower'],179,nothing)
cv2.createTrackbar('H Higher','mask',mask_hsv['H_Higher'],179,nothing)
cv2.createTrackbar('S Lower','mask',mask_hsv['S_Lower'],255,nothing)
cv2.createTrackbar('S Higher','mask',mask_hsv['S_Higher'],255,nothing)
cv2.createTrackbar('V Lower','mask',mask_hsv['V_Lower'],255,nothing)
cv2.createTrackbar('V Higher','mask',mask_hsv['V_Higher'],255,nothing)

cv2.namedWindow('circle')
cv2.createTrackbar('Min Radius','circle',circle_params['min_radius'],255,nothing)
cv2.createTrackbar('Max Radius','circle',circle_params['max_radius'],255,nothing)
cv2.createTrackbar('param1','circle',circle_params['param1'],255,nothing)
cv2.createTrackbar('param2','circle',circle_params['param2'],255,nothing)
cv2.createTrackbar('dp','circle',circle_params['param3'],255,nothing)
cv2.createTrackbar('minDist','circle',circle_params['param4'],255,nothing)
cv2.setTrackbarMin('param1', 'circle', 1)
cv2.setTrackbarMin('param2', 'circle', 1)
cv2.setTrackbarMin('dp', 'circle', 1)
cv2.setTrackbarMin('minDist', 'circle', 1)

while(1):
    _,img = camera.read()
    img = cv2.flip(img,1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red_hL = cv2.getTrackbarPos('H Lower','hsv')
    red_hH = cv2.getTrackbarPos('H Higher','hsv')
    red_sL = cv2.getTrackbarPos('S Lower','hsv')
    red_sH = cv2.getTrackbarPos('S Higher','hsv')
    red_vL = cv2.getTrackbarPos('V Lower','hsv')
    red_vH = cv2.getTrackbarPos('V Higher','hsv')
    mask_hL = cv2.getTrackbarPos('H Lower','mask')
    mask_hH = cv2.getTrackbarPos('H Higher','mask')
    mask_sL = cv2.getTrackbarPos('S Lower','mask')
    mask_sH = cv2.getTrackbarPos('S Higher','mask')
    mask_vL = cv2.getTrackbarPos('V Lower','mask')
    mask_vH = cv2.getTrackbarPos('V Higher','mask')
    minRadius = cv2.getTrackbarPos('Min Radius','circle')
    maxRadius = cv2.getTrackbarPos('Max Radius','circle')
    param1 = cv2.getTrackbarPos('param1','circle')
    param2 = cv2.getTrackbarPos('param2','circle')
    param3 = cv2.getTrackbarPos('dp','circle')
    param4 = cv2.getTrackbarPos('minDist','circle')

    red_hsv = {
        "H_Lower" : red_hL,
        "H_Higher" : red_hH,
        "S_Lower" : red_sL,
        "S_Higher" : red_sH,
        "V_Lower" : red_vL,
        "V_Higher" : red_vH,
    }

    mask_hsv = {
        "H_Lower" : mask_hL,
        "H_Higher" : mask_hH,
        "S_Lower" : mask_sL,
        "S_Higher" : mask_sH,
        "V_Lower" : mask_vL,
        "V_Higher" : mask_vH,
    }

    circle_params = {
        "min_radius" : minRadius,
        "max_radius" : maxRadius,
        "param1" : param1,
        "param2" : param2,
        "param3" : param3,
        "param4" : param4,
    }

    with open("red_hsv.json", "w") as outfile:
        json.dump(red_hsv, outfile)
        
    with open("mask_hsv.json", "w") as outfile:
        json.dump(mask_hsv, outfile)

    with open("circle_params.json", "w") as outfile:
        json.dump(circle_params, outfile)

    kernel = np.ones((7,7),"uint8")

    #red
    red_LowerRegion = np.array([red_hL,red_sL,red_vL],np.uint8)
    red_upperRegion = np.array([red_hH,red_sH,red_vH],np.uint8)

    red_object = cv2.inRange(hsv,red_LowerRegion,red_upperRegion)

    red_object = cv2.morphologyEx(red_object,cv2.MORPH_OPEN,kernel)
    red_object = cv2.dilate(red_object,kernel,iterations=1)

    red_object = cv2.erode(red_object, None, iterations=2)
    red_object = cv2.dilate(red_object, None, iterations=2)

    red_edges = cv2.Canny(red_object, 50, 150)

    
    #mask
    mask_LowerRegion = np.array([mask_hL,mask_sL,mask_vL],np.uint8)
    mask_upperRegion = np.array([mask_hH,mask_sH,mask_vH],np.uint8)

    mask_object = cv2.inRange(hsv,mask_LowerRegion,mask_upperRegion)

    mask_object = cv2.morphologyEx(mask_object,cv2.MORPH_OPEN,kernel)
    mask_object = cv2.dilate(mask_object,kernel,iterations=1)

    mask_object = cv2.erode(mask_object, None, iterations=2)
    mask_object = cv2.dilate(mask_object, None, iterations=2)

    # mask_edges = cv2.Canny(mask_object, 50, 150)

    circles = cv2.HoughCircles(red_edges, cv2.HOUGH_GRADIENT, param3, param4,
                              param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    
    red_and_mask = cv2.bitwise_and(red_object, mask_object)

    output=cv2.bitwise_and(img, img, mask = red_object)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # print(circles)
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("Mask",output)
    cv2.imshow("Edge", red_edges)
    # cv2.imshow("Ball detect ",circles)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        camera.release()
        cv2.destroyAllWindows()
        break