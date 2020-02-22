import cv2
import numpy as np
import pickle
import os
import imutils
from PIL import Image
import time

def get_hog():
    winSize = (32,32)
    blockSize = (8,8)
    blockStride = (4,4)
    cellSize = (4,4)
    nbins = 9
    derivAperture = 1
    winSigma = -1
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels,signedGradient)
    return hog
    affine_flags = cv2.WARP_INVERSE_MAP|cv.INTER_LINEAR

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, confidence, prob, x, y, x_plus_w, y_plus_h):
    
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0, 255, 255), 2)
    cv2.putText(img,'plateScore: %.2f%%'%(confidence*100),(2,60),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(img,'objectScore: %.2f%%'%(prob*100),(2,90),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    
hog = get_hog()

clf = pickle.loads(open("D:/HCMUT/HK191/ComputerVision/BTL/SVM/output/clf.pkl", "rb").read())
le = pickle.loads(open("D:/HCMUT/HK191/ComputerVision/BTL/SVM/output/le.pkl", "rb").read())


image = cv2.imread("D:/HCMUT/HK191/ComputerVision/BTL/Data/testCar/002.jpg")
start = time.time()
Width = image.shape[1]
Height = image.shape[0]
scale = 0.001
classes = None
with open("car.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet("car.weights", "car.cfg")
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
net.setInput(blob)
outs = net.forward(get_output_layers(net))
class_ids = []
confidences = []
prob = []
boxes = []
conf_threshold = .3

nms_threshold = .5
    
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            prob.append(detection[4])
            boxes.append([x, y, w, h])

# apply non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for j in indices:
    j = j[0]
    box = boxes[j]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    
    draw_bounding_box(image, confidences[j], prob[j], round(x), round(y), round(x+w), round(y+h))

    img = image[round(y):round(y+h), round(x):round(x+w)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("thresh", thresh)
    contours, k = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    Ca = []
    selcnts=[]
    boxx = []
    hog_descriptor = []
    pn = []
    text = []
    
    for i in range(len(contours)):
        boxx.append(cv2.boundingRect(contours[i]))
    for i in range(len(boxx)-1):
        for j in range(i+1, len(boxx)):
            if(boxx[i][0] > boxx[j][0]):
                inc = boxx[i]
                boxx[i] = boxx[j]
                boxx[j] = inc;
    for i in range(len(boxx)):
        (xx, yy, ww, hh) = boxx[i]
        if( hh > ww and ww/hh > 0.17 and hh/h > 0.35):
            text.append(boxx[i])
    for i in range(len(text)):
        (xx, yy, ww, hh) = text[i]
        if(i==0):
            dis = 1
            www = 0
        else:
            dis = xx - text[i-1][0]
            www = text[i-1][2]
        if(dis - www > 0):
            #cv2.rectangle(img,(xx,yy),(xx+ww,yy+hh),(0,255,0),2)            
            data = thresh[yy:yy + hh, xx:xx + ww]
            #cv2.imshow("char{}".format(i), data)
            img_bound = cv2.resize(data,(32,32))
            hog_descriptor= hog.compute(img_bound)
            hog_descriptor=np.squeeze(hog_descriptor)
            hog_descriptor=np.array([hog_descriptor,hog_descriptor])
            prediction = clf.predict_proba(hog_descriptor)[0]
            k = np.argmax(prediction)
            proba = prediction[k]
            name = le.classes_[k]
            pn.append(name)
    k = 0    
    for i in range(len(pn)):
        cv2.putText(image,pn[i],(10+k,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        k = k + 18
stop = time.time()
print("Time: ", (stop-start))
cv2.imshow("Img", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
