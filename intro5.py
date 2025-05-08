import cv2
import numpy as np 

#image
image = cv2.imread("assets/blobs.jpeg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur_image = cv2.blur(gray, (5,5))

detected_circles = cv2.HoughCircles(blur_image, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 1, maxRadius = 40)
if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    for pt in detected_circles[0,:]:
        a, b, r = pt[0], pt[1], pt[2]
        cv2.circle(image, (a,b), r, (0, 255, 0), 2)
        cv2.circle(image, (a,b), 1, (0,0,255), 2)
        cv2.imshow("Detected Circles", image)
        cv2.waitKey(0)
cv2.destroyAllWindows()

image = cv2.imread("assets/blobs.jpeg", 0)

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.9
params.filterByConvexity = True
params.minConvexity = 0.2
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(image)

blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0,0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Number of Circular Blobs:" + str(number_of_blobs)
cv2.putText(blobs, text, (20,550), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 100, 255), 2)

cv2.imshow("Filtering Circular Blobs Only", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()