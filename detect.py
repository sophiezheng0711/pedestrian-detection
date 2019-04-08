import numpy as np
import cv2 as cv

# TODO inputs (maybe do the cmdline thing in the tutorial?)
imgPath = "val (2).jpg"
confidenceThreshold = 0.5
nmsThreshold = 0.3

# TODO training & path
configPath = "/Users/sophiezheng/Desktop/pedestrian-detection/yolov3.cfg"
weightsPath = "/Users/sophiezheng/Desktop/pedestrian-detection/yolov3.weights"

net = cv.dnn.readNetFromDarknet(configPath, weightsPath)

image = cv.imread(imgPath)
(H, W) = image.shape[:2]

# TODO Supposedly we only want "People" to be detected, how is this going to help?
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
# start = time.time()
layerOutputs = net.forward(ln)
# end = time.time()

# bounding boxes around the object after detection
boxes = []
# confidence values assigned to object. Objects with lower confidence values might not be what the network thinks it is.
confidences = []
# detected object's class label (in our case, we only need "People")
classIDs = []

# TODO may not need this many layers? Maybe only need "People"
for lO in layerOutputs:
    # Loop over detections
    for detect in lO:
        # extract information
        scores = detect[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # compare to threshold to filter out detected objects with low confidence values
        if confidence > confidenceThreshold:
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
