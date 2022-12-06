import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

frame = cv.imread("Frame_img.jpg")
gray_clr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
height, width = gray_clr.shape

intergal_image = np.zeros((height, width))

for i in range(height):
    for j in range(width):
        intergal_image[i][j] = int(gray_clr[i][j])

for i in range(1, width):
    intergal_image[0][i] += intergal_image[0][i - 1]

for j in range(1, height):
    intergal_image[j][0] += intergal_image[j - 1][0]

for i in range(1, height):
    for j in range(1, width):
        intergal_image[i][j] = intergal_image[i - 1][j] + intergal_image[i][j - 1] - intergal_image[i - 1][j - 1] + gray_clr[i][j]

print(intergal_image)
np.savetxt('integral_matrix.txt', intergal_image, fmt='%d')
plt.plot(intergal_image)
plt.savefig("integral_image.jpg")
plt.show()