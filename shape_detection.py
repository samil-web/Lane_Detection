import cv2

img = cv2.imread('test_images/solidWhiteCurve.jpg', cv2.IMREAD_GRAYSCALE)


img = cv2.GaussianBlur(img, (5,5), 0)

ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

cv2.imshow('Threshold', thresh)
cv2.imshow('Contours', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

