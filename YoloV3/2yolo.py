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
from collections import OrderedDict
import torch
from tensorflow.keras.preprocessing.image import img_to_array#图片转为array

import pywt
import pywt.data

import logging
		
		
if __name__ == '__main__': #防止調用的時候這一行被運行
	
    data = r"Caries"
    data_trn = data + r"/Train"
    data_tst = data + r"/Test"
    img = r"/Image"
    lbl = r"/Label"
    ori = r"/Ori"

    datapath = r"../Caries/Train"
	
    ctrn = open(data_trn + r"/caries_train.txt", "w")
    cval = open(data_trn + r"/caries_val.txt", "w")

	
    trnimg = glob.glob(data_trn + img + "/*")
    timg = tqdm(trnimg)
    timg.set_description("list trn img")
    for ti in timg:
        filepath, full_filename = os.path.split(ti)
        ctrn.write(str(datapath) + str(img) + "/" +str(full_filename) + "\n")
			
			
    valimg = glob.glob(data_tst + img + "/*")
    vallbl = glob.glob(data_tst + lbl + "/*")
    vimg = tqdm(valimg)
    vlbl = tqdm(vallbl)
    vimg.set_description("list & copy val img")
    vlbl.set_description("copy val lbl")
    for vi in vimg:
        filepath, full_filename = os.path.split(vi)
        cval.write(str(datapath) + str(img) + "/" +str(full_filename) + "\n")
        shutil.copyfile((vi), (data_trn + img + "/" + full_filename))
    for vl in vlbl:
        filepath, full_filename = os.path.split(vl)
        shutil.copyfile((vl), (data_trn + lbl + "/" + full_filename))

    ctrn.close()
    cval.close()

    trnlbl = glob.glob(data_trn + lbl + "/*")
    tlbl = tqdm(trnlbl)
    tlbl.set_description("copy trn lbl")
    for tl in tlbl:
        filepath, full_filename = os.path.split(tl)
        shutil.copyfile((tl), (data_trn + img + "/" + full_filename))
	

