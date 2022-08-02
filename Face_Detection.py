'''
    Date: 5-5-2021
    Name: Engineer-D
    Idea: OpenCV face detection
    Special Thanks: PyImageSearch
'''

# Import the necessary packages
import numpy as np
import argparse
import cv2

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, \
    help = "path to input image")
ap.add_argument("-p", "--prototxt", required = True, \
    help = "path to caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required = True,\
    help = "path to caffe pre-trained model")
ap.add_argument("-c", "--confidence", type = float, default=0.5,\
    help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load Our Serialized model from disk
print("[INFO] Loading Model ...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args['image'])
#print(image.shape[:2])    ============> (420, 320)
(h,w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0,\
    (300, 300), (104.0, 177.0, 123.0))

# Pass the blob through the network and obtain the detections and \
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

# Loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the \
    # prediction
    confidence = detections[0, 0, i, 2]

    # Filter out weak detection by ensuring the confidence is
    # greater than the minimum confidence
    if confidence > args["confidence"]:
        # Compute the (x, y) -coordinates of the bounding box for
        # the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX,startY), (endX,endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),\
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
