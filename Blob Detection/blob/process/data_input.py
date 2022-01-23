#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array#图片转为array
from tensorflow.keras.utils import to_categorical#相当于one-hot
from imutils import paths
import cv2
import numpy as np
import random
import os
import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# 此处的二元分类，可以不需要one_hot编译，np.random.randint可以直接生成0 1 编码
# Example: y_train = np.random.randint(2,size=(1000,1))

import logging
# log的部分
logging.basicConfig(level='INFO',
                    format='[%(asctime)s %(levelname)-8s %(module)-12s:Line %(lineno)-3d] %(message)s ',
                    datefmt='[%H:%M:%S]')

log = logging.getLogger('data_input.py')

def load_cvdata(cv_path):
    image = []#数据x
    label = []#标签y
    cariesnum = 0
    normalnum = 0
	
    cvdata_paths = glob.glob(cv_path + "/*")
    if (len(cvdata_paths) <= 0): log.warning("cvdata_paths is empty\n")

    log.info("running function \"load_cv_data\" \n")
    t = tqdm(cvdata_paths) # Create progress bar
    t.set_description("loaded cv data")
    for each_path in t:
        filepath, filename = os.path.split(each_path)
        normal = glob.glob(each_path + "/00000/*")
        caries = glob.glob(each_path + "/00001/*")

        if (len(normal) > 0): 
            if (len(caries) <= 0) and (random.random() > 0.1): 
                image.append(filename)
                label.append(0)
                normalnum += 1
            elif (len(caries) > 0) :
                image.append(filename)
                label.append(1)
                cariesnum += 1

    print("> caries  :" + str(cariesnum))
    print("> normal:" + str(normalnum))
    label = np.array(label)
    label = to_categorical(label,num_classes=2)#one-hot
    return image,label
	
def color_adjust(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image
	
def load_data(path, norm_size, class_num, predict = 0):
    #log.info("running function \"load_data\" with path %s \n" % str(path))
    data = []#数据x
    label = []#标签y
    image_paths = sorted(list(paths.list_images(path)))#imutils模块中paths可以读取所有文件路径
    if (len(image_paths) <= 0): log.warning("%s is empty\n" % str(path))
    random.seed(1)#保证每次数据顺序一致
    random.shuffle(image_paths)#将所有的文件路径打乱
    predict_list = []
    #t = tqdm(image_paths) # Create progress bar
    #t.set_description("loaded data")
    #for each_path in t:
    for each_path in image_paths:
        image = cv2.imread(each_path)#读取文件
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        oriimage = image.copy()
        image = cv2.resize(image,(norm_size,norm_size))#统一图片尺寸
        image = (image - image.mean())/image.std() #標準化
        image = img_to_array(image)
		
        maker = int(each_path.split(os.path.sep)[-2])#切分文件目录，类别为文件夹整数变化，从0-61.如train文件下00014，label=14
        if maker == 0 and random.random() > 0.92:
            label.append(maker)
            data.append(image)
        elif maker == 1:
            label.append(maker)
            data.append(image)
        if predict == 1:
            filepath, full_filename = os.path.split(each_path)
            filename, suffix = os.path.splitext(full_filename)
            filename = filename.strip("(")
            filename = filename.strip(")")
            location = tuple(map(int, filename.split(', ')))
            predict_list.append((maker,location,image,oriimage))
    data = np.array(data,dtype="float")
    label = np.array(label)
    label = to_categorical(label,num_classes=class_num)#one-hot
    if predict == 0: 
        return data, label
    else:
        return predict_list
        
