import cv2
import matplotlib.pyplot as plt

# PART 1
# read the image
image = cv2.imread("../img/heart/fourcameracut.jpg")
image = cv2.blur(image, (25, 25))
# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# create a binary thresholded image
_, binary = cv2.threshold(gray, 64, 64, cv2.THRESH_BINARY_INV)
cv2.imwrite('../img/heart_testpackage/black-white-snapshot.png', binary)
# show it
# plt.imshow(binary, cmap="gray")
# plt.show()


# PART 2

# find the contours from the thresholded image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw all contours
image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
cv2.imwrite('../img/heart_testpackage/color-snapshot.png', image)
# show the image with the drawn contours

plt.imshow(image)
plt.show()
