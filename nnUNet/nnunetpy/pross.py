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
from skimage import morphology
from skimage.feature import blob_log, hessian_matrix_eigvals, hessian_matrix
from skimage.filters import hessian
from scipy.linalg import sqrtm
from math import sqrt
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
from shapely.geometry.polygon import Polygon
import shutil
from collections import OrderedDict
import torch
from tensorflow.keras.preprocessing.image import img_to_array#图片转为array
from torch.nn.parallel import DistributedDataParallel

import pywt
import pywt.data

import logging

def convert_bool_uint8(image, n=0):
    #將image值map到0和255並轉成uint8
    image = image > n #mix all type caries to boolean
    image = image *255 #convert boolean to 0/255
    image = image.astype(np.uint8)
    return image
		
		
if __name__ == '__main__': #防止調用的時候這一行被運行

    '''
    url = "http://dentaltw-info.uc.r.appspot.com/labels/completed"
    response = urllib.request.urlopen(url)
    tm_text = json.loads(response.read())
    with open('data.json','w') as f:
        json.dump(tm_text, f)
    '''
    split = r"splithalf"
    split_Tr = split + r"/training"
    split_Ts = split + r"/testing"
    split_mask = r"/output"
    split_image = r"/input"
    split_ori = r"/oris"
	
    
    oExist = True
    while os.path.exists(split):
        os.rename(split, split + r"_delete")
        if oExist:
            shutil.rmtree(split + r"_delete")
            oExist = False
    os.mkdir(split)

    for dist in [split_Tr,split_Ts]:
        os.mkdir(dist)
        for ddist in [split_mask,split_image,split_ori]:
            os.mkdir(dist+ddist)
    

    with open( "data.json") as f:
        tm_text = json.load(f)
	
    numdata = 20000
    skip = 0

    for key,value in tqdm(tm_text.items()):
        if skip:
            skip = 0
            continue
			
        record = {}
        raw_data = tm_text[key]
		
        if(raw_data['filename'] == None):
            continue
        filename = raw_data['filename']
		
        oimg = io.imread(filename)
        height, width = oimg.shape[:2]
		
        url,filename = os.path.split(str(filename))
        filename, suffix = os.path.splitext(filename)

        if int(filename) >= numdata: 
            break
        elif int(filename) == 267 or int(filename) == 38:
            skip = 1#268 bug

        record["image_id"] = key
        record["height"] = height
        record["width"] = width
		
        cariescheck = 0
        caries = np.zeros((height, width),np.uint8)
		
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
					
            else:
                continue

        savenormalpic = False
        if (cariescheck == 0): 
            if random.random() > 0.99: 
                savenormalpic = True
            else: 
                continue

        if random.random() > 0.5: #reduce cuda memory test
            continue

        #cuth,cutw = int(height*0.05),int(width*0.05)
        cuth,cutw = int(height*0),int(width*0)
		
        caries = convert_bool_uint8(caries) #將caries值map到0和255並轉成uint8
        caries = caries[cuth:height-cuth, cutw:width-cutw]
        oimg = oimg[cuth:height-cuth, cutw:width-cutw]
        overlay = oimg.copy()
		
        #將caries部分上紅色,這部分先完成避免框的顔色改變
        bimg = overlay[:,:,0]
        gimg = overlay[:,:,1]
        rimg = overlay[:,:,2]
        bimg[caries == 255] = 0
        gimg[caries == 255] = 0
        rimg[caries == 255] = 255
        overlay[:,:,0] = bimg
        overlay[:,:,1] = gimg
        overlay[:,:,2] = rimg
        coimg = oimg - (oimg*0.4).astype(np.uint8) + (overlay *0.4).astype(np.uint8)
		
        if savenormalpic: 		
            cv2.imwrite( split_Ts+split_image+"/" +str(filename)+".jpg", oimg)
            cv2.imwrite( split_Ts+split_mask+"/" +str(filename)+".jpg", caries)
            cv2.imwrite( split_Ts+split_ori+"/" +str(filename)+".jpg", coimg)
            savenormalpic = False
            continue
			
        if random.random() > 0.1 : 
            cv2.imwrite( split_Tr+split_image+"/" +str(filename)+".jpg", oimg)
            cv2.imwrite( split_Tr+split_mask+"/" +str(filename)+".jpg", caries)
            cv2.imwrite( split_Tr+split_ori+"/" +str(filename)+".jpg", coimg)
        else :
            cv2.imwrite( split_Ts+split_image+"/" +str(filename)+".jpg", oimg)
            cv2.imwrite( split_Ts+split_mask+"/" +str(filename)+".jpg", caries)
            cv2.imwrite( split_Ts+split_ori+"/" +str(filename)+".jpg", coimg)

