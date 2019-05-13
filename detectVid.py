import argparse
import numpy as np
import cv2 as cv
import imutils
import time
import os

argp = argparse.ArgumentParser()
argp.add_argument("-vid", "--video", required=True, help="path to input image")
arg = vars(argp.parse_args())

# inputs, including image, threshold values, config file, and weights file
confidenceThreshold = 0.5
nmsThreshold = 0.3

configPath = "yolov3.cfg"
weightsPath = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = cv.VideoCapture(arg["video"])
writer = None
(H, W) = (None, None)

while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDS = []

    for lO in layerOutputs:
        for detect in lO:
            scores = detect[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidenceThreshold and classID == 0:
                box = detect[0:4] * np.array([W, H, W, H])
                (cX, cY, w, h) = box.astype("int")

                x = int(cX - (w / 2))
                y = int(cY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                classIDS.append(classID)

    idxs = cv.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = (255, 0, 0)
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if writer is None:
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter("out.mp4", fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)

writer.release()
vs.release()
