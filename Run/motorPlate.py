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
    cv2.putText(img,'plateScore: %.2f%%'%(confidence*100),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(img,'objectScore: %.2f%%'%(prob*100),(x,y_plus_h+20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    
    
hog = get_hog()

clf = pickle.loads(open("D:/HCMUT/HK191/ComputerVision/BTL/SVM/output/clf.pkl", "rb").read())
le = pickle.loads(open("D:/HCMUT/HK191/ComputerVision/BTL/SVM/output/le.pkl", "rb").read())

start = time.time()

image = cv2.imread("D:/HCMUT/HK191/ComputerVision/BTL/Data/test/Motor200.jpg")
Width = image.shape[1]
Height = image.shape[0]
scale = 0.0009
classes = None
with open("motor.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet("motor.weights", "motor.cfg")
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
    up = round(h*2/3)
    down = round(h/3)
    
    draw_bounding_box(image, confidences[j], prob[j], round(x), round(y), round(x+w), round(y+h))

    img = image[round(y):round(y+h), round(x):round(x+w)]
    upbox = image[round(y):round(y+up), round(x):round(x+w)]
    downbox = image[round(y+down):round(y+h), round(x):round(x+w)]
    grayup = cv2.cvtColor(upbox, cv2.COLOR_BGR2GRAY)
    graydown = cv2.cvtColor(downbox, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray,(3,3),0)
    ret1, threshup = cv2.threshold(grayup,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret2, threshdown = cv2.threshold(graydown,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    cntup, kup = cv2.findContours(threshup, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cntdown, kdown = cv2.findContours(threshdown, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('thrup', threshup)
    cv2.imshow('thrdown', threshdown)
    

    selcnts=[]
    upboxx = []
    downboxx = []
    hog_descriptor = []
    pnup = []
    pndown = []
    textup = []
    textdown = []
    areaup = []
    areadown = []
    for i in range(len(cntup)):
        areaup.append(cv2.contourArea(cntup[i]))
        upboxx.append(cv2.boundingRect(cntup[i]))

    for i in range(len(upboxx)-1):
        for j in range(i+1, len(upboxx)):
            if(upboxx[i][0] > upboxx[j][0]):
                inc = upboxx[i]
                inc1 = areaup[i]
                areaup[i] = areaup[j]
                areaup[j] = inc1
                upboxx[i] = upboxx[j]
                upboxx[j] = inc;
                
    for i in range(len(upboxx)):
        (xx, yy, ww, hh) = upboxx[i]
        if( hh > ww and areaup[i] > 20 and areaup[i] < 200 and ww/hh > .1 and ww/hh < .4):
            textup.append(upboxx[i])
    for i in range(len(textup)):
        (xx, yy, ww, hh) = textup[i]
        if(i==0):
            dis = 1
            www = 0
        else:
            dis = xx - textup[i-1][0]
            www = textup[i-1][2]
        if(dis - www > 0):
            #cv2.rectangle(upbox,(xx,yy),(xx+ww,yy+hh),(0,255,0),2)            
            data = threshup[yy:yy + hh, xx:xx + ww]
            cv2.imshow("aaa{}".format(i), data)
            img_bound = cv2.resize(data,(32,32))
            hog_descriptor= hog.compute(img_bound)
            hog_descriptor=np.squeeze(hog_descriptor)
            hog_descriptor=np.array([hog_descriptor,hog_descriptor])
            prediction = clf.predict_proba(hog_descriptor)[0]
            k = np.argmax(prediction)
            proba = prediction[k]
            name = le.classes_[k]
            pnup.append(name)
 ###################################################333333           
    for i in range(len(cntdown)):
        areadown.append(cv2.contourArea(cntdown[i]))
        downboxx.append(cv2.boundingRect(cntdown[i]))
        
    for i in range(len(downboxx)-1):
        for j in range(i+1, len(downboxx)):
            if(downboxx[i][0] > downboxx[j][0]):
                inc = downboxx[i]
                inc1 = areadown[i]
                areadown[i] = areadown[j]
                areadown[j] = inc1
                downboxx[i] = downboxx[j]
                downboxx[j] = inc;
                
    for i in range(len(downboxx)):
        (xx, yy, ww, hh) = downboxx[i]
        if( hh > ww and areadown[i] > 20 and areadown[i] < 200 and ww/hh > .1 and ww/hh < .4):
            textdown.append(downboxx[i])
    for i in range(len(textdown)):
        (xx, yy, ww, hh) = textdown[i]
        if(i==0):
            dis = 1
            www = 0
        else:
            dis = xx - textdown[i-1][0]
            www = textdown[i-1][2]
        if(dis - www > 0):
            #cv2.rectangle(downbox,(xx,yy),(xx+ww,yy+hh),(0,255,0),2)            
            data = threshdown[yy:yy + hh, xx:xx + ww]
            cv2.imshow("char{}".format(i), data)
            img_bound = cv2.resize(data,(32,32))
            hog_descriptor= hog.compute(img_bound)
            hog_descriptor=np.squeeze(hog_descriptor)
            hog_descriptor=np.array([hog_descriptor,hog_descriptor])
            prediction = clf.predict_proba(hog_descriptor)[0]
            k = np.argmax(prediction)
            proba = prediction[k]
            name = le.classes_[k]
            pndown.append(name)
    k = 0    
    for i in range(len(pnup)):
        cv2.putText(image,pnup[i],(10+k,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        k = k + 18
    k = 0
    for i in range(len(pndown)):
        cv2.putText(image,pndown[i],(10+k,60),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        k = k + 18
stop = time.time()
print("Time: ", (stop-start))
cv2.imshow("Img", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
