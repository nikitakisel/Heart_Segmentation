import cv2 as cv
img = cv.imread("img/heart_testpackage/black-white-snapshot.png")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Applying SIFT detector
sift = cv.SIFT_create()
kp = sift.detect(gray, None)

# Marking the keypoint on the image using circles
img = cv.drawKeypoints(gray,
                        kp,
                        img,
                        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imwrite('img/heart_testpackage/image-with-keypoints.jpg', img)


ready_img = cv.imread("img/heart_testpackage/image-with-keypoints.jpg")
cv.imshow("Display window", ready_img)
k = cv.waitKey(0)
