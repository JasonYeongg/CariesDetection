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

def extract_bboxes(masks):
    bboxes = []
    for mask in masks:
        # print(mask.shape)
        m = GenericMask(mask.to("cpu").numpy(), mask.shape[0], mask.shape[1])
        bboxes.append(m.bbox())
    return bboxes

def convert_bool_uint8(image, n=0):
    #將image值map到0和255並轉成uint8
    image = image > n #mix all type caries to boolean
    image = image *255 #convert boolean to 0/255
    image = image.astype(np.uint8)
    return image

def sharpen(img, sigma=100):    
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

    return usm
	
def json_handle(outdir_ori,outdir_train,outdir_val,outdir_cv,numdata):

    acctxt = open('prossacc.txt','w')

    with open( "data.json") as f:
        tm_text = json.load(f)
		
    '''
    trainp = glob.glob(outdir_train+"/*")
    valp = glob.glob(outdir_val+"/*")
    traindata = []
    valdata = []
    
    t = tqdm(trainp) # Create progress bar
    t.set_description("loaded train")
    for c in t:
        filepath, full_filename = os.path.split(c)
        filename, suffix = os.path.splitext(full_filename)
        traindata.append(filename)
		
    t = tqdm(valp) # Create progress bar
    t.set_description("loaded val")
    for n in t:
        filepath, full_filename = os.path.split(n)
        filename, suffix = os.path.splitext(full_filename)
        valdata.append(filename)	
    '''
	
    dataset_dictst = []
    dataset_dictsv = []
	
    skip = 0
    checkedp = 0
    emptyp = 0

    for key,value in tqdm(tm_text.items()):
        if skip :
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
			
        #if filename not in np.hstack([traindata,valdata]):
            #continue
		
        record["image_id"] = key
        record["height"] = height
        record["width"] = width
		
        cariescheck = 0
        caries = np.zeros((height, width),np.uint8)
        teeth = np.zeros((height, width),np.uint8)
		
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
				
                #if (np.count_nonzero(temp)/(width*height)) >= 0.055:
                    #continue
                #if (int(np.min(px))/int(np.max(px))) <= 0.04:
                    #continue

                caries = caries + temp
					
            else:
                continue
				
        if (cariescheck == 0): 
            continue
			
        outdir_f = outdir_cv + r"/" + filename
        Exist = True
        while os.path.exists(outdir_f):
            os.rename(outdir_f, outdir_f + r"_delete")
            if Exist:
                shutil.rmtree(outdir_f + r"_delete"); Exist = False
        os.mkdir(outdir_f)
        os.mkdir(outdir_f + r"/00000"); os.mkdir(outdir_f + r"/00001");
		
        '''
        savenormalpic = False
        if (cariescheck == 0): 
            if random.random() > 0.99: 
                savenormalpic = True
            else: 
                continue
        '''
		
        cuth,cutw = int(height*0.05),int(width*0.05)
		
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
        cv2.imwrite( outdir_ori + "/" +str(filename)+".jpg", coimg)
		
        oimg = cv2.cvtColor(oimg, cv2.COLOR_BGR2GRAY)
        oimg = sharpen(oimg,sigma=100)

        ero = morphology.erosion(oimg, morphology.square(3))
        dil = morphology.dilation(ero, morphology.square(3))
		
        th = cv2.threshold(dil, 245, 255, cv2.THRESH_BINARY)[1]
		
        mix=dil.copy()
        mix[th==255]=255
		
        mix = cv2.GaussianBlur(mix, (3,3), 0)#noise reduction
		
        blobs_log = blob_log(1-mix, num_sigma=5, threshold=.1)

        black = np.zeros((oimg.shape[1],oimg.shape[0]),dtype=np.uint8)
        mix2 = cv2.cvtColor(mix, cv2.COLOR_GRAY2RGB)
        for blob in blobs_log:
            y, x, std = blob
            r,y,x = int(std * sqrt(2)),int(y),int(x)
            if(r>=3 and x-r>0 and y-r>0 and x+r<oimg.shape[1] and y+r<oimg.shape[0]):
                mix2 = cv2.circle(mix2,(int(x), int(y)),r,(0,255,255),3)

                #black = cv2.circle(black,(int(x), int(y)),r,(100),-1)
				
                check = caries[(int(y-r)):(int(y+r)), (int(x-r)):(int(x+r))]
                checky,checkx = check.shape
                cut = int(checky * 0.1)
				
                #'''
                if  (np.count_nonzero(check[cut:(checky-cut), cut:(checkx-cut)])) > 0: #cut white frame to reduce unnecessary blob
                    cv2.imwrite(outdir_f + r"/00001/(" + str(y)  + ", " + str(x)  + ", " + str(r)  + ").jpg", oimg[(int(y-r)):(int(y+r)), (int(x-r)):(int(x+r))])
                    #cv2.imwrite(outdir_f + r"/00001/(" + str(y)  + ", " + str(x)  + ", " + str(r)  + ").jpg", tht[(int(y-r)):(int(y+r)), (int(x-r)):(int(x+r))])
                else:
                    cv2.imwrite(outdir_f + r"/00000/(" + str(y)  + ", " + str(x)  + ", " + str(r)  + ").jpg", oimg[(int(y-r)):(int(y+r)), (int(x-r)):(int(x+r))])
                    #cv2.imwrite(outdir_f + r"/00000/(" + str(y)  + ", " + str(x)  + ", " + str(r)  + ").jpg", tht[(int(y-r)):(int(y+r)), (int(x-r)):(int(x+r))])
                #'''

        #plt.imshow((np.hstack((coimg,mix2))),cmap='gray')
        #plt.show()
        #cv2.imwrite( outdir_ori + "/" +str(filename)+"blob.jpg", mix2) #------------------------------------------------------------------------------------
		
        #oimg = cv2.equalizeHist(adjusted)
		
        #check cover rate
        cnts = []
        cnts = cv2.findContours(caries, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#caries輪廓
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for cc in cnts:
            if (len(cc) < 3):
                continue
            ccc = np.zeros((oimg.shape[1],oimg.shape[0]),np.uint8)
            cv2.fillPoly(ccc, [cc], 100)

            pchecksize = np.count_nonzero((black + ccc) >= 150)
		
            if  int(pchecksize) > 0:
                checkedp += 1
            else:
                emptyp += 1
		
        acctxt.write(str(checkedp) + "   " + str(emptyp) + "\n")
		
        cv2.imwrite( outdir_ori + "/" +str(filename)+"blob.jpg", mix2)
        #print(checkedp,emptyp)

        '''
        if savenormalpic: 		
            record["file_name"] = outdir_val + "/" + filename+".jpg"
            dataset_dictsv.append(record)
            cv2.imwrite( outdir_val + "/" +str(filename)+".jpg", oimg) 
			
            savenormalpic = False
            continue
		
		
        if  random.random() > 0.09: 
        #if  filename in traindata: 
            record["file_name"] = outdir_train +"/" + filename+".jpg"
            cv2.imwrite( outdir_train + "/" +str(filename)+".jpg", oimg) 
			
        else: 
        #elif  filename in valdata: 
            record["file_name"] = outdir_val + "/" + filename+".jpg"
            cv2.imwrite( outdir_val + "/" +str(filename)+".jpg", oimg) 
			
        #else: 
            #continue
        '''
		
    acctxt.write('\n \n \n \n')
    acctxt.write("checkedp: " + str(checkedp) + "\n")
    acctxt.write("emptyp: " + str(emptyp) + "\n")
    acctxt.write('\n')
    acctxt.close()
		
if __name__ == '__main__': #防止調用的時候這一行被運行

    outdir = r"data"
    outdir_ori = outdir + r"/ori"
    outdir_train = outdir + r"/train"
    outdir_val = outdir + r"/val"
    outdir_cv = outdir + r"/cv"
	
    #'''
    oExist = True
    while os.path.exists(outdir):
        os.rename(outdir, outdir + r"_delete")
        if oExist:
            shutil.rmtree(outdir + r"_delete")
            oExist = False
    os.mkdir(outdir)

    for dist in [outdir_ori,outdir_train,outdir_val,outdir_cv]:
        Exist = True
        while os.path.exists(dist):
            os.rename(dist, dist + r"_delete")
            if Exist:
                shutil.rmtree(dist + r"_delete")
                Exist = False
        os.mkdir(dist)
    #'''
	
    json_handle(outdir_ori,outdir_train,outdir_val,outdir_cv,9000)#2000
