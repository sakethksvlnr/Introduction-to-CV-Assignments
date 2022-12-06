import cv2
import numpy as np

image1 = cv2.imread("files/marker1.png")
image2 = cv2.imread("files/marker2.png")
Z = 800


x, y, w, h = [100,200,50,1]
cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 5)

Ip1x = (x + (x+w))//2
Ip1Y = (y + (y+h))//2

'''mask_red_2 = cv2.inRange(image2, lower_red, upper_red)
contours, _ = cv2.findContours(mask_red_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)'''

'''for c in contours:
    rect = cv2.boundingRect(c)
    if rect[2] < 1 or rect[3] < 1:
        continue'''
x, y, w, h = [200,300,100,1]
cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 5)

Ip2x = (x + (x+w))//2
Ip2Y = (y + (y+h))//2

dist = 450

focal = 685

D = round((dist * focal)/abs(Ip1x - Ip2x), 2)

print("Distance: {} mm".format(D))
cv2.waitKey(0)