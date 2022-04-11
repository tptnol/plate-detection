# Harvey Villanueva
# 4/6/22

'''
Used to extract plate images from a picture.
Inspired by 
https://towardsdatascience.com/extracting-regions-of-interest-from-images-dacfd05a41ba
'''

# USAGE (command prompt)
# python plate_detection.py --image assets/image.jpeg

# imports
import argparse
import cv2
from cv2 import findContours
import numpy as np
import contour_lib # library of common functions for countour segmentation

# constants
RESIZE_IMAGE = True
MIN_CONTOUR_AREA = 7700
DEBUG = False

# passing the image in through cmd prompt
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
    help="path to input image")
args = vars(ap.parse_args())
filename = args["image"]
# img = cv2.imread('assets/plate.jpeg', 0) # initially converting to grayscale 

cv2.destroyAllWindows()
img = cv2.imread(filename)
(H, W) = img.shape[:2]
print('Image Shape', (H, W))

# resizing the image, for ease of looking at
if RESIZE_IMAGE:
    img = cv2.resize(img, (0, 0), fx = 0.25, fy = 0.25)
    (H, W) = img.shape[:2]
    print('New Image Shape', img.shape)
clone = img.copy()

img_centre_x = W//2
img_centre_y = H//2
print("Image centre = ({},{})".format(img_centre_x, img_centre_y))

# original
cv2.imshow("Original", img)

# grayscale image
if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)

# gaussian blur
ksize = 7 # represents the size of the kernel
blur = cv2.GaussianBlur(gray, (ksize, ksize), 0) 
blur = cv2.medianBlur(gray, ksize) # remove salt and pepper noise
cv2.imshow("Blurred", blur)

# canny edge detection
lower = 60 # represents lower tail threshold for hysteresis procedure
higher = 120 # represents lower tail threshold for hysteresis procedure
canny = cv2.Canny(blur, lower, higher)
cv2.imshow("Canny", canny)

# dilate and erode over multiple iterations
'''
dilateSize = 3
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilateSize, dilateSize))
canny_morphed = cv2.dilate(canny, kernel, iterations=5)
canny_morphed = cv2.erode(canny, kernel, iterations=5)
cv2.imshow("Canny Morphed", canny_morphed)
'''

# extract contours
contours_list = contour_lib.get_contours(canny, min_contour_area=0)
print("Number of contours = " + str(len(contours_list)))

#print(contours_list)

# black background with contours on top
blackbg = np.zeros(clone.shape)
for c in contours_list:
  cv2.drawContours(blackbg, c[0], 0, (0,255,0), cv2.FILLED)
  cv2.imshow("Filled Contours with Black Background", blackbg)

'''
# black background with mask
mask = blackbg[:, :, 0].astype("uint8")
roi_black = cv2.bitwise_and(clone, clone, mask=mask)
cv2.imshow("ROI with Black Background", roi_black) 
'''

# basically the opposites --> white background

'''
# white background with contours on top
whitebg = np.ones(clone.shape)
for c in contours_list:
  cv2.drawContours(whitebg, c[0], 0, (0,0,0), cv2.FILLED)
  cv2.imshow("Filled Contours with White Background", whitebg)
'''

'''
# white background with mask
mask = whitebg[:,:,0].astype("uint8")
white_roi = cv2.add(roi_black, whitebg.astype("uint8"))
cv2.imshow("ROI with White Background", white_roi)
'''

cv2.waitKey(0)
cv2.destroyAllWindows()
