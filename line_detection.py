import cv2
import numpy as np
import math

def region(image):
    height, width = image.shape
    triangle = np.array([
                       [(0, height+100), (width//2, round(height/1.75)), (width, height)]
                       ])
    
    mask = np.zeros_like(image)
    
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def polygon_region(lines, image):
    
    lines = lines.squeeze()
    left_bottom_idx = lines[:, 0].argmin(axis=0)
    x0_left = lines[left_bottom_idx, 0]
    y0_left = lines[left_bottom_idx, 1]

    right_bottom_idx = lines[:, 0].argmax(axis=0)
    x0_right = lines[right_bottom_idx,0]
    y0_right = lines[right_bottom_idx,1]

    left_top_idx = lines[lines[:, 2] < image.shape[0]//2][:, 2].argmax(axis=0)
    x1_left = lines[left_top_idx, 2]
    y1_left = lines[left_top_idx, 3]

    right_top_idx = lines[lines[:, 2] > image.shape[0]//2][:, 2].argmin(axis=0)
    x1_right = lines[right_top_idx, 2]
    y1_right = lines[right_top_idx, 3]

    polygon = np.array([
                       [(x0_left,y0_left), (x1_left,y1_left), (x1_right, y1_right), (x0_right, y0_right)]
                       ])
    
    mask = np.zeros_like(image)
    
    mask = cv2.fillPoly(mask, polygon, 255)
    mask = cv2.bitwise_and(image, mask)

    print((x0_left, y0_left), (x1_left, y1_left), (x0_right,y0_right), (x1_right, y1_right))
    return mask

img = cv2.imread('test_images/solidYellowCurve.jpg', cv2.IMREAD_GRAYSCALE)
original = img.copy()


triangle = region(img)

img = cv2.GaussianBlur(triangle, (5,5), 0)
ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

canny = cv2.Canny(img, 50, 150, 3)
edges = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
edgesP = edges.copy()

lines = cv2.HoughLines(canny, 0.9, np.pi/180, threshold=50, min_theta=0, max_theta=np.pi)

for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(edges, pt1, pt2, (0,0,255), 3, cv2.LINE_AA) 


linesP = cv2.HoughLinesP(canny, 1, np.pi / 180, 20, minLineLength=0, maxLineGap=10)

for i in range(0, len(linesP)):
     l = linesP[i][0]
     cv2.line(edgesP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

polygon = polygon_region(linesP, original)
cv2.imshow('Original', img)
# cv2.imshow('Triangle', triangle)
# cv2.imshow('Canny', canny)
# cv2.imshow('Lines', edges)
cv2.imshow('Lines probabilistic', edgesP)
cv2.imshow('Polygon', polygon)

cv2.waitKey(0)
cv2.destroyAllWindows()













# def points_to_region(image, lines):

#     points = np.concatenate((lines.squeeze()[:, 0:2], lines.squeeze()[:, 2:]), axis=0)
#     polygon = np.asarray([list(map(tuple, points))])

#     print(polygon)
#     mask = np.zeros_like(image)
    
#     mask = cv2.fillPoly(mask, polygon, 255)
#     mask = cv2.bitwise_and(image, mask)

    
#     return mask

