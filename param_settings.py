import cv2
import numpy as np
import json
import os.path
from os import path

camera = cv2.VideoCapture(0)

def nothing(x):
    pass

hsv_empty = {
    "H_Lower" : 0,
    "H_Higher" : 0,
    "S_Lower" : 0,
    "S_Higher" : 0,
    "V_Lower" : 0,
    "V_Higher" : 0,
}
circle_empty = {
    "min_radius" : 0,
    "max_radius" : 0,
    "param1" : 0,
    "param2" : 0,
    "param3" : 0,
    "param4" : 0,
}
canny_empty = {
    "min_radius" : 0,
    "max_radius" : 0,
    "param1" : 0,
    "param2" : 0,
    "param3" : 0,
    "param4" : 0,
}
object_hsv_path = 'object_hsv.json'
mask_hsv_path = 'mask_hsv.json'
circle_params_path = 'circle_params.json'
canny_params_path = 'canny_params.json'

if path.exists(object_hsv_path):
    with open(object_hsv_path, 'r') as openfile:
        object_hsv = json.load(openfile)
else:
    object_hsv = hsv_empty

if path.exists(mask_hsv_path):
    with open(mask_hsv_path, 'r') as openfile:
        mask_hsv = json.load(openfile)
else:
    mask_hsv = hsv_empty

if path.exists(circle_params_path):
    with open(circle_params_path, 'r') as openfile:
        circle_params = json.load(openfile)
else:
    circle_params = circle_empty


cv2.namedWindow('hsv red')
cv2.createTrackbar('H Lower','hsv',object_hsv['H_Lower'],179,nothing)
cv2.createTrackbar('H Higher','hsv',object_hsv['H_Higher'],179,nothing)
cv2.createTrackbar('S Lower','hsv',object_hsv['S_Lower'],255,nothing)
cv2.createTrackbar('S Higher','hsv',object_hsv['S_Higher'],255,nothing)
cv2.createTrackbar('V Lower','hsv',object_hsv['V_Lower'],255,nothing)
cv2.createTrackbar('V Higher','hsv',object_hsv['V_Higher'],255,nothing)

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

    #object
    object_hL = cv2.getTrackbarPos('H Lower','hsv')
    object_hH = cv2.getTrackbarPos('H Higher','hsv')
    object_sL = cv2.getTrackbarPos('S Lower','hsv')
    object_sH = cv2.getTrackbarPos('S Higher','hsv')
    object_vL = cv2.getTrackbarPos('V Lower','hsv')
    object_vH = cv2.getTrackbarPos('V Higher','hsv')
    
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

    object_hsv = {
        "H_Lower" : object_hL,
        "H_Higher" : object_hH,
        "S_Lower" : object_sL,
        "S_Higher" : object_sH,
        "V_Lower" : object_vL,
        "V_Higher" : object_vH,
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

    with open(object_hsv_path, "w") as outfile:
        json.dump(object_hsv, outfile)
        
    with open(mask_hsv_path, "w") as outfile:
        json.dump(mask_hsv, outfile)

    with open("circle_params.json", "w") as outfile:
        json.dump(circle_params, outfile)

    kernel = np.ones((7,7),"uint8")

    #red
    object_LowerRegion = np.array([object_hL,object_sL,object_vL],np.uint8)
    object_upperRegion = np.array([object_hH,object_sH,object_vH],np.uint8)

    object_mask = cv2.inRange(hsv,object_LowerRegion,object_upperRegion)

    object_mask = cv2.morphologyEx(object_mask,cv2.MORPH_OPEN,kernel)
    object_mask = cv2.dilate(object_mask,kernel,iterations=1)

    object_mask = cv2.erode(object_mask, None, iterations=2)
    object_mask = cv2.dilate(object_mask, None, iterations=2)

    object_edges = cv2.Canny(object_mask, 50, 150)

    
    #mask
    mask_LowerRegion = np.array([mask_hL,mask_sL,mask_vL],np.uint8)
    mask_upperRegion = np.array([mask_hH,mask_sH,mask_vH],np.uint8)

    field_mask = cv2.inRange(hsv,mask_LowerRegion,mask_upperRegion)

    field_mask = cv2.morphologyEx(field_mask,cv2.MORPH_OPEN,kernel)
    field_mask = cv2.dilate(field_mask,kernel,iterations=1)

    field_mask = cv2.erode(field_mask, None, iterations=2)
    field_mask = cv2.dilate(field_mask, None, iterations=2)

    # mask_edges = cv2.Canny(field_mask, 50, 150)

    circles = cv2.HoughCircles(object_edges, cv2.HOUGH_GRADIENT, param3, param4,
                              param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    
    object_and_mask = cv2.bitwise_and(object_mask, field_mask)

    output=cv2.bitwise_and(img, img, mask = object_mask)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # print(circles)
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("Mask",output)
    cv2.imshow("Edge", object_edges)
    # cv2.imshow("Ball detect ",circles)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        camera.release()
        cv2.destroyAllWindows()
        break