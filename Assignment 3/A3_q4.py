import cv2
import numpy as np
import time
import math
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

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 256  # Width of network's input image
inpHeight = 256  # Height of network's input image
start = time.time()

# Load names of classes
classesFile = "files/coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# modelConfiguration = "tiny-yolov2-trial13.cfg"
# modelWeights = "tiny-yolov2-trial13.weights"

modelConfiguration = "files/yolov3-tiny.cfg"
modelWeights = "files/yolov3-tiny.weights"

# modelConfiguration = "yolov3.cfg"
# modelWeights = "yolov3.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    outputlayers = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return outputlayers


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    #print(boxes)
    left, top, width, height = 0, 0 , 0, 0
    if len(indices) > 0:
        for i in indices:
            print(i)
            box = boxes[i[0]]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv2.putText(frame, classes[classIds[i[0]]], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(round(confidences[i[0]] * 100, 2)) + "%", (left, top + height + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 3)



# Process inputs

# cap = cv2.VideoCapture(0)
counter = 0
time_elasped = 0

with dai.Device(pipeline,usb2Mode=True) as device:
    # Output queue will be used to get the disparity frames from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    while cv2.waitKey(1) < 0:

        # get frame from the video
        # hasFrame, frame = cap.read()
        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()
        counter += 1

        visualize = frame.copy()

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        postprocess(frame, outs)
        cv2.putText(frame, "Q to Exit", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
        time_elasped = int(time.time() - start)
        if time_elasped > 1:
            cv2.putText(frame, "FPS: " + str(counter // time_elasped), (50, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (255, 255, 255), 3)
            print("FPS: ", counter // time_elasped)

        frame = cv2.resize(frame, (720, 480))

        cv2.imshow("Object Detection YOLO", frame)

        # Stop the program if reached end of video
        if cv2.waitKey(1) == ord('q'):
            print("Done processing !!!")
            # cap.release()
            end = time.time()
            print("Time Elasped: ", int(end - start))
            print("FPS: ", counter // (end - start))
            break
