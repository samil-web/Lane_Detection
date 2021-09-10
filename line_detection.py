import cv2
import numpy as np
import math

def region(image):
    height, width = image.shape
    triangle = np.array([
                       [(0, height), (width//2, round(height/1.75)), (width, height)]
                       ])
    
    mask = np.zeros_like(image)
    
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

img = cv2.imread('test_images/solidYellowCurve.jpg', cv2.IMREAD_GRAYSCALE)


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

print(linesP)

for i in range(0, len(linesP)):
     l = linesP[i][0]
     cv2.line(edgesP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)


# cv2.imshow('Triangle', triangle)
# cv2.imshow('Canny', canny)
# cv2.imshow('Lines', edges)
cv2.imshow('Lines probabilistic', edgesP)

cv2.waitKey(0)
cv2.destroyAllWindows()
