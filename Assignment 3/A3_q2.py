import cv2
import numpy as np


reference_image = cv2.imread('files/reference.png',cv2.IMREAD_GRAYSCALE)
image = cv2.imread('files/image.png',cv2.IMREAD_GRAYSCALE)

reference_image = np.array(reference_image).astype(np.float32)
image = np.array(image).astype(np.float32)

kernel_x = np.array([[-1., 1.], [-1., 1.]]) * .25
kernel_y = np.array([[-1., -1.], [1., 1.]]) * .25
kernel_t = np.array([[1., 1.], [1., 1.]]) * .25

# normalize pixels
reference_image = reference_image / 255.
image = image / 255.

imageX = np.array(cv2.filter2D(reference_image, -1, kernel=kernel_x))
imageY = np.array(cv2.filter2D(reference_image, -1, kernel=kernel_y))
imageT = np.array(cv2.filter2D(reference_image, -1, kernel=kernel_t)) + np.array(cv2.filter2D(image, -1, kernel=kernel_x))

motion = np.divide(imageT, np.sqrt(np.square(imageX) + np.square(imageY)))
np.savetxt('motion.txt', motion, fmt='%f')
print(motion)

