import numpy as np
import cv2
import argparse
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt

#parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to image")
args = vars(ap.parse_args())

#load image and apply SEEDS
seeds = None
num_superpixels = 200
prior = 2
num_levels = 4
num_histogram_bins = 5
num_iterations = 4

img = cv2.imread(args["image"])
converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, channels = converted_img.shape

#create SEEDS object
seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, 
prior, num_histogram_bins)

#apply SEEDS to image
seeds.iterate(converted_img, num_iterations)

#retrieve segmentation result
labels = seeds.getLabels()

#show SEEDS output
fig = plt.figure("SEEDS Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(converted_img), labels))
plt.axis("off")
plt.show()

#loop over superpixels
for (i, segVal) in enumerate(np.unique(labels)):
    print "[x] inspecting segment %d" % (i)
    mask = np.zeros(converted_img.shape[:2], dtype="uint8")
    mask[labels == segVal] = 255
    
    #show masked region
    cv2.imshow("Mask", mask)
    cv2.imshow("Applied", cv2.bitwise_and(converted_img, converted_img, mask = mask))
    cv2.waitKey(0)
