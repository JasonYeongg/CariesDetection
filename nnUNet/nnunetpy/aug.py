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
from math import sqrt
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
from shapely.geometry.polygon import Polygon
import shutil
import torch
import logging

def convert_bool_uint8(image, n=0):
    #將image值map到0和255並轉成uint8
    image = image > n #mix all type caries to boolean
    image = image *255 #convert boolean to 0/255
    image = image.astype(np.uint8)
    return image

def rotate(image,label):

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.randrange(2, 16), 1.0)
    nimage = cv2.warpAffine(image, M, (w, h))
    nlabel = cv2.warpAffine(label, M, (w, h))
 
    return nimage,nlabel
	
def resize(image,label):

    (h, w) = image.shape[:2]
    size =  random.randrange(7, 9)
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
	
if __name__ == '__main__': #防止調用的時候這一行被運行

    pathname = r"Task201_Caries"
	
    for split in ["training","testing"]:
        image_paths = glob.glob(pathname + "/"+split+"/input/*")
        for t in tqdm(image_paths):
            if (str(t[-3:]) != "jpg"):
                continue
            image = io.imread(t)
            filepath, filename = os.path.split(t)
            filename = filename[:-4]
            label = io.imread(pathname + "/"+split+"/output/" + filename  + ".jpg")

            if random.random() > 0.3 :
                image,label = flip(image,label)
            if random.random() > 0.3 :
                image,label = rotate(image,label)
            if random.random() > 0.3 :
                image,label = resize(image,label)
		
            cv2.imwrite(pathname + "/"+split+"/input/aug" + str(filename)+".jpg", image)
            cv2.imwrite(pathname + "/"+split+"/output/aug" + str(filename)+".jpg", label)
		
