import numpy as np
import os, json, cv2, random, cv2, glob
import urllib.request
import PIL.Image as Image
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from skimage import io
from scipy.linalg import sqrtm
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from math import sqrt
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
from shapely.geometry.polygon import Polygon
import shutil
from collections import OrderedDict
import torch
from tensorflow.keras.preprocessing.image import img_to_array#图片转为array

import pywt
import pywt.data

import logging


def addon(overlay, caries , b, g, r):
    bimg = overlay[:,:,0]
    gimg = overlay[:,:,1]
    rimg = overlay[:,:,2]
    bimg[caries == 255] = b
    gimg[caries == 255] = g
    rimg[caries == 255] = r
    overlay[:,:,0] = bimg
    overlay[:,:,1] = gimg
    overlay[:,:,2] = rimg
    return overlay
	
def rotate(image,label):

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.randrange(2, 16), 1.0)
    nimage = cv2.warpAffine(image, M, (w, h))
    nlabel = cv2.warpAffine(label, M, (w, h))
 
    return nimage,nlabel
	
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), (255,),int(im.shape[0]*0.005))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), (255,),int(im.shape[0]*0.005))
	
def elastic(image, label):
	
    alpha = image.shape[1] * 1.2 #1.5
    sigma = image.shape[1] * 0.09 #0.08
    alpha_affine = image.shape[1] * 0.09 #0.08
	
    random_state = np.random.RandomState(None)

    shape = label.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    nlabel = map_coordinates(label, indices, order=1, mode='reflect').reshape(shape)	

    return image, nlabel
	
def resize(image,label):

    (h, w) = image.shape[:2]
    size =  random.randrange(6, 9)
    if random.random() > 0.5 :
        nimage = cv2.resize(image, (w,int(h*0.1*size)), interpolation=cv2.INTER_AREA)
        nlabel = cv2.resize(label, (w,int(h*0.1*size)), interpolation=cv2.INTER_AREA)
    else :
        nimage = cv2.resize(image, (int(w*0.1*size),h), interpolation=cv2.INTER_AREA)
        nlabel = cv2.resize(label, (int(w*0.1*size),h), interpolation=cv2.INTER_AREA)
 
    return nimage,nlabel

    (h, w) = image.shape[:2]
    size =  random.randrange(6, 9)
    if random.random() > 0.5 :
        nimage = cv2.resize(image, (w,int(h*0.1*size)), interpolation=cv2.INTER_AREA)
        nlabel = cv2.resize(label, (w,int(h*0.1*size)), interpolation=cv2.INTER_AREA)
    else :
        nimage = cv2.resize(image, (int(w*0.1*size),h), interpolation=cv2.INTER_AREA)
        nlabel = cv2.resize(label, (int(w*0.1*size),h), interpolation=cv2.INTER_AREA)
 
    return nimage,nlabel
	
def flip(image,label):
    if random.random() > 0.6 :
        if random.random() > 0.5 :
            nimage = cv2.flip(image, 0)
            nlabel = cv2.flip(label, 0)
        else :
            nimage = cv2.flip(image, 1)
            nlabel = cv2.flip(label, 1)
    else :
            nimage = cv2.flip(image, -1)
            nlabel = cv2.flip(label, -1)
    return nimage,nlabel
	
def dil(image,label):
    reducenum = random.randrange(10, 15) *  0.001
    reducenum2 = random.randrange(10, 15) *  0.001
    dkernel = np.ones((int(image.shape[0]*reducenum),int(image.shape[0]*reducenum)),dtype=np.uint8)
    ekernel = np.ones((int(image.shape[0]*reducenum*0.7),int(image.shape[0]*reducenum*0.7)),dtype=np.uint8)
    if random.random() > 0.4 :
        if random.random() > 0.6 :
            nlabel = cv2.dilate(label, dkernel)
        else:
            nlabel = cv2.dilate(label, np.ones((int(image.shape[0]*reducenum),int(image.shape[0]*reducenum2)),dtype=np.uint8))
    else :
        nlabel = cv2.erode(label, ekernel)
		 
    return image,nlabel


def convert_bool_uint8(image, n=0):
    #將image值map到0和255並轉成uint8
    image = image > n #mix all type caries to boolean
    image = image *255 #convert boolean to 0/255
    image = image.astype(np.uint8)
    return image
	
def xy2xywh(px,py,width,height):

    xmin,ymin,xmax,ymax = np.min(px), np.min(py), np.max(px), np.max(py)
	
    x = (xmin + (xmax-xmin)/2) * 1.0 / width
    y = (ymin + (ymax-ymin)/2) * 1.0 / height
    w = (xmax-xmin) * 1.0 / width
    h = (ymax-ymin) * 1.0 / height
	
    return x,y,w,h
	
def aug(image, label, img, lbl, ori, split, filename, i):

    augcheck = 0
    if split == r"Caries/Train":
        aug = r"/aug"
    else:
        aug = r"/vaug"
	
    if random.random() > 0.666 :
        image,label = rotate(image,label)
        augcheck += 1
    if random.random() > 0.666 :
        image,label = flip(image,label)
        augcheck += 1
    if random.random() > 0.666 :
        image,label = resize(image,label)
        augcheck += 1
    if random.random() > 0.75:
        image,label = dil(image,label)
        augcheck += 1
    if random.random() > 0.95:
        image,label = elastic(image, label)
        augcheck += 1
		
    if augcheck:
	
        lblt = open(split + lbl + aug + str(filename) + "-" + str(i) + r".txt", "w")
        rect = np.zeros((image.shape[0], image.shape[1]),np.uint8)
	
        cnts = []
        cnts = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#caries輪廓
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        if len(cnts) > 0:
            for cc in cnts:
                if (len(cc) < 3):
                    continue
                cch = np.hsplit(cc[:,-1],2)
                x,y,w,h = xy2xywh(cch[0],cch[1],image.shape[1], image.shape[0])
                lblt.write("0 " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) +  "\n")
                rect = cv2.rectangle(rect, (int((x-w/2)*image.shape[1]), int((y-h/2)*image.shape[0])), (int((x+w/2)*image.shape[1]), int((y+h/2)*image.shape[0])), 1, 2)

        cv2.imwrite(split + img + aug + str(filename) + "-" + str(i) + ".jpg",  image)
        lblt.close()
		
        rect = convert_bool_uint8(rect)
		
        overlay = image.copy()
        overlay = addon(overlay, label, 0, 0, 255)
        overlay = addon(overlay, rect, 0, 255, 0)
        coimg = image - (image*0.4).astype(np.uint8) + (overlay *0.4).astype(np.uint8)
        cv2.imwrite(r"Caries/Test" + ori + aug + str(filename) + "-" + str(i) + ".jpg", coimg)
		
		
if __name__ == '__main__': #防止調用的時候這一行被運行

    '''
    url = "http://dentaltw-info.uc.r.appspot.com/labels/completed"
    response = urllib.request.urlopen(url)
    tm_text = json.loads(response.read())
    with open('data.json','w') as f:
        json.dump(tm_text, f)
    '''
	
    data = r"Caries"
    data_trn = data + r"/Train"
    data_tst = data + r"/Test"
    img = r"/Image"
    lbl = r"/Label"
    ori = r"/Ori"
	
    #'''
    oExist = True
    while os.path.exists(data):
        os.rename(data, data + r"_delete")
        if oExist:
            shutil.rmtree(data + r"_delete")
            oExist = False
    os.mkdir(data)

    for dist in [data_trn,data_tst]:
        os.mkdir(dist)
        for ddist in [img,lbl]:
            os.mkdir(dist+ddist)
    os.mkdir(data_tst+ori)
    #'''

    with open( "data.json") as f:
        tm_text = json.load(f)
	
    skip = 0
    numdata = 20000
	
    imgskip = []
    spath = glob.glob(r"Skip/*")
    for element in spath:
        filepath, full_filename = os.path.split(element)
        filename, suffix = os.path.splitext(full_filename)
        imgskip.append(filename)

    for key,value in tqdm(tm_text.items()):
        if skip:
            skip=0
            continue
			
        cont = 0
		
        raw_data = tm_text[key]
		
        if(raw_data['filename'] == None):
            continue
        filename = raw_data['filename']
		
        oimg = io.imread(filename)
        height, width = oimg.shape[:2]
		
        url,filename = os.path.split(str(filename))
        filename, suffix = os.path.splitext(filename)
		
        if  random.random() > 0.15: 
            split = data_trn
        else:
            split = data_tst

        if int(filename) >= numdata: 
            break
        elif int(filename) == 267 or int(filename) == 38:
            skip = 1#268 bug
        elif filename in imgskip:
            continue
		
        cariescheck = 0
        caries = np.zeros((height, width),np.uint8)
        rect = np.zeros((height, width),np.uint8)
		
        cariesbox = []
		
        for anno in raw_data['regions']:
            if(anno == None):
                continue
            region = anno['region_attributes']

            temp = Image.new("L", [width, height,], 0)
		
            if 'detail' in region and region['detail'] == 'caries':
                cariescheck = 1
                px = anno['shape_attributes']['all_points_x']
                py = anno['shape_attributes']['all_points_y']
                if(len(px) < 3):
                    continue
					
                combined = list(map(tuple, np.vstack((px, py)).T))
                ImageDraw.Draw(temp).polygon(combined, outline=1, fill=1)
                temp = np.array(temp)
                temp = convert_bool_uint8(temp)
				
                if (np.count_nonzero(temp)/(width*height)) >= 0.1:
                    continue
                if (int(np.min(px))/int(np.max(px))) <= 0.04:
                    continue

                caries = caries + temp
				
                x,y,w,h = xy2xywh(px,py,width,height)
                cariesbox.append((x,y,w,h))
                
                rect = cv2.rectangle(rect, (int((x-w/2)*width), int((y-h/2)*height)), (int((x+w/2)*width), int((y+h/2)*height)), 1, 2)
					
            else:
                continue
		
        if cariesbox :
            lblt = open(split + lbl + "/" + str(filename) + r".txt", "w")
            for x,y,w,h in cariesbox:
                lblt.write("0 " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) +  "\n")
            lblt.close()
        else :
            if (split == data_tst and random.random() > 0.95): 
                lblt = open(split + lbl + "/" + str(filename) + r".txt", "w")
                lblt.close()
            else :
                continue
				
        caries = convert_bool_uint8(caries)
		
        if cariesbox :
            if split == data_trn:
                for i in range(1,random.randrange(1, 26)):
                    aug(oimg, caries, img, lbl, ori, split, filename, i)
            else :
                for i in range(1,random.randrange(1, 11)):
                    aug(oimg, caries, img, lbl, ori, split, filename, i)
					
        rect = convert_bool_uint8(rect) 
        overlay = oimg.copy()
		
        #將caries部分上紅色,這部分先完成避免框的顔色改變
        overlay = addon(overlay, caries, 0, 0, 255)
        overlay = addon(overlay, rect, 0, 255, 0)
        coimg = oimg - (oimg*0.4).astype(np.uint8) + (overlay *0.4).astype(np.uint8)

        cv2.imwrite(split + img + "/" +str(filename)+ ".jpg", oimg)
        if split == data_trn :
            cv2.imwrite(data_tst + ori + "/" +str(filename)+ ".jpg", coimg)
        else :
            cv2.imwrite(data_tst + ori + "/val" +str(filename)+ ".jpg", coimg)

