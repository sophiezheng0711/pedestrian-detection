import os

os.chdir("darknet/")
os.system("./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights")
