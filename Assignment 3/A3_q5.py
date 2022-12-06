import cv2
import depthai as dai


DIM = (720, 480)

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True

# Create pipeline
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.video.link(xoutRgb.input)

haar_cascade_file = 'files/Haarcascade_frontalface_default.xml'

with dai.Device(pipeline,usb2Mode=True) as device:
    # Output queue will be used to get the disparity frames from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    while True:

        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()
        frame = cv2.resize(frame, DIM, interpolation=cv2.INTER_AREA)

        # Converting image to grayscale
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Loading the required haar-cascade xml classifier file
        haar_cascade = cv2.CascadeClassifier(haar_cascade_file)

        # Applying the face detection method on the grayscale image
        faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)

        # Iterating through rectangles of detected faces
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("rgb", frame)
        if cv2.waitKey(1) == ord('q'):
            break
