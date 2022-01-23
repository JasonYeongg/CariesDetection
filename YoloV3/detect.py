import os, json, cv2, random, cv2, glob
import argparse
import sys
import numpy as np
import pandas as pd
import os.path
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
from skimage import io

# Initialize the parameters
#basThreshold = 0.01  #Basic threshold
#confThreshold = 0.5  #Confidence threshold
#nmsThreshold = 0.5  #Non-maximum suppression threshold

inpWidth = 608  #608     #Width of network's input image
inpHeight = 608 #608     #Height of network's input image

version = "Caries6"
path = version + "/Test"
image_paths = glob.glob(path + "/Image/*")
ori_paths = path + "/Ori/"
        
# Load names of classes
classesFile =  version + "/classes.names"

classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

modelConfiguration = r"/home/dentall/Desktop/jason_test/yolov3/"+version+"/darknet-yolov3.cfg"
modelWeights = r"/home/dentall/Desktop/jason_test/yolov3/"+version+"/weights/darknet-yolov3_final.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

def count(Result):
    TP,FN,IP,FP = Result
    sens = TP/(TP+FN)
    prec = TP/(TP+FP)
    f1 = (2*sens*prec)/(sens+prec)
	
    sens = round(sens*100,2)
    prec = round(prec*100,2)
    f1 = round(f1*100,2)
	
    return sens,prec,f1

def convert2one(image, n=0):
    #將image值map到0和255並轉成uint8
    image = image > n #mix all type caries to boolean
    image = image * 1 #convert boolean to 0/255
    image = image.astype(np.uint8)
    return image

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	
def intersection_over_union(frame1, frame2):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(frame1[0], frame2[0]) #求兩個窗口左上角x座標最大值 
    yA = max(frame1[1], frame2[1]) #求兩個窗口左上角y座標最大值 
    xB = min(frame1[2], frame2[2]) #求兩個窗口右下角x座標最小值 
    yB = min(frame1[3], frame2[3]) #兩個窗口右下角y座標最小值 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    frame1Area = (frame1[2] - frame1[0] +1 ) * (frame1[3] - frame1[1] +1)
    frame2Area = (frame2[2] - frame2[0] +1 ) * (frame2[3] - frame2[1] +1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(frame1Area + frame2Area - interArea)
    #iou = interArea / float(frame1Area)
    # return the intersection over union value
    return iou

def drawPred(frame, classId, conf, left, top, right, bottom,R,G,B):
    cv2.rectangle(frame, (left, top), (right, bottom), (R,G,B), 2)

    label = '%.2f' % conf
        
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    txtsize = ((top-bottom) * 0.008)
    if(txtsize<0):
        txtsize*=-1
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, txtsize, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]*0.7), top + baseLine), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, txtsize, (0,0,0), 2)

def process(frame, outs, basThreshold):

    #dbox = np.zeros((frame.shape[0],frame.shape[1]),np.uint8)
    dbox = []
	
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:

            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
			
            center_x = int(detection[0] * frameWidth)
            center_y = int(detection[1] * frameHeight)
            width = int(detection[2] * frameWidth)
            height = int(detection[3] * frameHeight)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
			
            if confidence > basThreshold:
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, left + width, top + height])
                drawPred(frame, classId, confidence, left, top, left + width, top + height,0,255,0)
                dbox.append((left, top, (left + width), (top + height)))
                #cv2.rectangle(dbox, (left, top), (left + width, top + height), 255, -1)
		
    return frame, dbox, classIds, confidences, boxes
	
def postprocess(frame2, classIds, confidences, boxes, confThreshold, nmsThreshold):

    dbox2 = []
    dbox3 = []
    frame3 = frame2.copy()

    clssnms = []
    confnms = []
    bboxnms = []
	
    for clss, conf, bbox in zip (classIds, confidences, boxes):
        if conf > confThreshold:
			
            drawPred(frame2, clss, conf, bbox[0], bbox[1], bbox[2], bbox[3],255,255,0)
            dbox2.append((bbox[0], bbox[1],bbox[2], bbox[3]))
			
            clssnms.append(clss)
            confnms.append(float(conf))
            bboxnms.append((bbox[0], bbox[1], bbox[2], bbox[3] ))
			
    #sortlist.sort(key=lambda x:x[0], reverse=True)


    indices = cv2.dnn.NMSBoxes(bboxnms, confnms, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        bbox = bboxnms[i]
		
        drawPred(frame3, clssnms[i], confnms[i], bbox[0], bbox[1], bbox[2], bbox[3],0,255,255)
        #cv2.rectangle(dbox3, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)
        dbox3.append((bbox[0], bbox[1],bbox[2], bbox[3]))
	
    return frame2, dbox2, frame3, dbox3
	
def checkansw(dbox,check,Result):
    TP,FP,IP,EP = Result
    #pred = []
    #pred = cv2.findContours(dbox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #pred = pred[0] if len(pred) == 2 else pred[1]
    answ = []
    answ = cv2.findContours(check, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    answ = answ[0] if len(answ) == 2 else answ[1]
	
    #dbox = convert2one(dbox)
    cbox = np.zeros((check.shape[0],check.shape[1]),np.uint8)
    check = convert2one(check)

    for p in dbox:
        peach = np.zeros((check.shape[0],check.shape[1]),np.uint8)
        cv2.rectangle(peach, (p[0], p[1]), (p[2], p[3]), 1, -1)
        cv2.rectangle(cbox, (p[0], p[1]), (p[2], p[3]), 1, -1)
		
        #peach = np.zeros((check.shape[0],check.shape[1]),np.uint8)
        #cv2.fillPoly(peach, [p], 1)

        pcorrect = np.count_nonzero((peach + check) >=2)
		
        if  int(pcorrect) > (np.count_nonzero(peach) * 0.1):
            TP += 1
        else:
            FP += 1
			
    for a in answ:
        if (len(a) < 3):
            continue
        aeach = np.zeros((check.shape[0],check.shape[1]),np.uint8)
        cv2.fillPoly(aeach, [a], 1)

        ainclude = np.count_nonzero((aeach + cbox) >=2)
		
        if  int(ainclude) > (np.count_nonzero(aeach) * 0.1):
            IP += 1
        else:
            EP += 1
			
    Result = TP,FP,IP,EP
    return Result


dtt = version + r"/Test/Dtt"
dtt2 = version + r"/Test/Dtt2"
dtt3 = version + r"/Test/Dtt3"
	
for d in [dtt,dtt2,dtt3]:
    oExist = True
    while os.path.exists(d):
        os.rename(d, d + r"_delete")
        if oExist:
            shutil.rmtree(d + r"_delete")
            oExist = False
    os.mkdir(d)


Result1 = [0,0,0,0]
Result2 = [0,0,0,0]
Result3 = [0,0,0,0]

Result23 = [0,0,0,0]
Result33 = [0,0,0,0]
Result43 = [0,0,0,0]

for t in tqdm(image_paths):

    if (str(t[-3:]) != "jpg"):
        continue
    image = io.imread(t)
    filepath, full_filename = os.path.split(t)
    filename = full_filename[:-4]

    if full_filename[:4] == 'vaug':
        continue
		
    check = np.zeros((image.shape[0],image.shape[1]),np.uint8)
	
    if os.stat(path + "/Label/" + filename  + ".txt").st_size != 0:
        labeltxt = []
        with open(path + "/Label/" + filename  + ".txt", 'r') as f:
            for line in f.readlines():
                labeltxt.append(line.split(' '))
        for label in labeltxt:
            cc,cx,cy,cw,ch = label
            cc,cx,cy,cw,ch = float(cc),float(cx),float(cy),float(cw),float(ch)
            cv2.rectangle(check, (int((cx-cw/2)*image.shape[1]), int((cy-ch/2)*image.shape[0])), (int((cx+cw/2)*image.shape[1]), int((cy+ch/2)*image.shape[0])), 255, -1)
            cv2.rectangle(image, (int((cx-cw/2)*image.shape[1]), int((cy-ch/2)*image.shape[0])), (int((cx+cw/2)*image.shape[1]), int((cy+ch/2)*image.shape[0])), (255,80,80), 2)

    blob = cv2.dnn.blobFromImage(image, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net.setInput(blob)

    outs = net.forward(getOutputsNames(net))

    imagecpy = image.copy()
    image, dbox, classIds, confidences, boxes = process(image, outs, 0.01)
    image2, dbox2, image3, dbox3 = postprocess(imagecpy, classIds, confidences, boxes, 0.5, 0.5)
	
    image22, dbox22, image23, dbox23 = postprocess(imagecpy, classIds, confidences, boxes, 0.4, 0.5)
    image32, dbox32, image33, dbox33 = postprocess(imagecpy, classIds, confidences, boxes, 0.3, 0.5)
    image42, dbox42, image43, dbox43 = postprocess(imagecpy, classIds, confidences, boxes, 0.2, 0.5)

    cv2.imwrite(dtt + "/" +str(filename)+ ".jpg", image.astype(np.uint8))
    cv2.imwrite(dtt2 + "/" +str(filename)+ ".jpg", image2.astype(np.uint8))
    cv2.imwrite(dtt3 + "/" +str(filename)+ ".jpg", image3.astype(np.uint8))
	
    Result1 = checkansw(dbox,check,Result1)
    Result2 = checkansw(dbox2,check,Result2)
    Result3 = checkansw(dbox3,check,Result3)
    Result23 = checkansw(dbox23,check,Result23)
    Result33 = checkansw(dbox33,check,Result33)
    Result43 = checkansw(dbox43,check,Result43)
	
print(Result1,count(Result1))
print(Result2,count(Result2))
print(Result3,count(Result3))
print(Result23,count(Result23))
print(Result33,count(Result33))
print(Result43,count(Result43))

