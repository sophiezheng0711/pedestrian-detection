import numpy as np
import cv2 as cv

# inputs, including image, threshold values, config file, and weights file
imgPath = "doll.jpg"
confidenceThreshold = 0.5
nmsThreshold = 0.3

configPath = "/Users/sophiezheng/Desktop/pedestrian-detection/yolov3.cfg"
weightsPath = "/Users/sophiezheng/Desktop/pedestrian-detection/yolov3.weights"

net = cv.dnn.readNetFromDarknet(configPath, weightsPath)

# import image and normalize data
image = cv.imread(imgPath)
(H, W) = image.shape[:2]

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(ln)

# bounding boxes around the object after detection
boxes = []
# confidence values assigned to object. Objects with lower confidence values might not be what the network thinks it is.
confidences = []
# detected object's class label (in our case, we only need "People")
classIDs = []

for lO in layerOutputs:
    # Loop over detections
    for detect in lO:
        # extract information
        scores = detect[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # compare to threshold to filter out detected objects with low confidence values
        if confidence > confidenceThreshold and classID == 0:
            box = detect[0:4] * np.array([W, H, W, H])
            (cX, cY, w, h) = box.astype("int")

            x = int(cX - (w / 2))
            y = int(cY - (h / 2))

            # add extracted information to output lists
            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# Applying non-maxima suppression suppresses significantly overlapping bounding boxes,
# keeping only the most confident ones. NMS also ensures that we don't have redundant or extraneous boxes
idxs = cv.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold)

# drawing the boxes of detected objects
if len(idxs) > 0:

    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = (255, 0, 0)
        cv.rectangle(image, (x, y), (x + w, y + h), color, 2)

# show the output image
cv.imshow("Image", image)
cv.waitKey(0)
