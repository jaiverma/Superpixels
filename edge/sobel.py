import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../images/frame.jpg', 0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)
sobel = np.absolute(sobel_x) + np.absolute(sobel_y)
#sobel = sobel_x + sobel_y
plt.subplot(2, 2, 1), plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(sobel_x, cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(sobel, cmap = 'gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])

plt.show()
