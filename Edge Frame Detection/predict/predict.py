#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import precision_recall_curve
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import sys
sys.path.append("../process")#添加其他文件夹
import datahandlet
import data_input
from skimage import io
import shutil
import json
from tqdm import tqdm
from PIL import Image, ImageDraw
import seaborn as sn
import urllib
import logging
# log的部分
logging.basicConfig(level='INFO',
                    format='[%(asctime)s %(levelname)-8s %(module)-12s:Line %(lineno)-3d] %(message)s ',
                    datefmt='[%H:%M:%S]')

log = logging.getLogger('predict.py')


def confusion_matrix(TP,TN,FP,FN, modelnum, save_dir='', names=["normal","caries"], normalize=True):
	
    matrix = np.zeros((2,2))
    matrix[0, 0] = TN
    matrix[0, 1] = FN
    matrix[1, 0] = FP
    matrix[1, 1] = TP
    array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
    fig = plt.figure(figsize=(12, 9), tight_layout=True)
    sn.set(font_scale=1.0)  # for label size
    labels = (0 < len(names) < 99) and len(names) == 2  # apply names to ticklabels
    sn.heatmap(array, annot = True, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
        xticklabels=names if labels else "auto",
        yticklabels=names if labels else "auto").set_facecolor((1, 1, 1))
    fig.axes[0].set_xlabel('True')
    fig.axes[0].set_ylabel('Predicted')
    fig.savefig(save_dir+"confusion_matrix"+str(modelnum+1)+".png", dpi=250)
	
    return array


channel = 1
height = 64
width = 64
class_num = 2
norm_size = 64  # 参数tf.enable_eager_execution()

outdir = r"../result/preresult"
outdir_nms = r"../result/nms"
def create_colormap():
    colormap_true = np.zeros((20,256,3), dtype = np.uint8)
    for color in range(256):
        if color <= 127: #N color
            colormap_true[:, color, 0] = 255
            colormap_true[:, color, 1] = int(color * 1.5)
            colormap_true[:, color, 2] = int(color * 1.5)
        else: #P color
            colormap_true[:, color, 0] = int((255 - color) * 1.5)
            colormap_true[:, color, 1] = 255
            colormap_true[:, color, 2] = 255

    return colormap_true

def read_K_index():
    train_str = []
    test_str = []
    with open(r'../model/K_index.txt', 'r') as K_index:
        K_index.readline() #讀取batch size
        K_index.readline() #讀取epochs
        K = int(K_index.readline()) #讀取K
        K_index.readline() #讀取換行
        for k in range(K):
            train_str.append(K_index.readline())
            test_str.append(K_index.readline())
            K_index.readline() #讀取換行
        K_index.readline() #讀取Order字
        K_order = []
        while True:
            line = K_index.readline()
            if line: K_order.append(line.strip())
            else: break
    for i, elem in enumerate(train_str):
        train_str[i] = np.fromstring(elem, dtype=int, sep=',')
    for i, elem in enumerate(test_str):
        test_str[i] = np.fromstring(elem, dtype=int, sep=',')
    return train_str, test_str, K_order, K

colormap_true = create_colormap()
train_index, test_index, K_order, K = read_K_index()

#自動生成file dependencis
for directory in [outdir, outdir_nms]:
    maked = False
    while not os.path.exists(directory):
        if not maked:
            log.warning("%s does not exist, creating...\n" % directory)
            os.mkdir(directory); maked = True

for k in range(K): #对每次kfold的split结果做loop
    #記錄test case全部預測結果
    TP=0
    TN=0
    FP=0
    FN=0
	
    TPs=0
    TNs=0
    FPs=0
    FNs=0
    model_k = "model%s.h5" % str(k+1)
    log.info("predicting %s validation data set" % model_k)
    model = keras.models.load_model(model_k) #加载模型
    for directory in [outdir, outdir_nms]:
        Exist = True
        while os.path.exists(directory + r"/%s" % str(k+1)):
            os.rename(directory + r"/%s" % str(k+1), directory + r"/%s" % str(k+1) + r"_delete")
            if Exist:
                log.warning("%s exist, will remove then create it...\n" % \
                            (directory + r"/%s" % str(k+1)))
                shutil.rmtree(directory + r"/%s" % str(k+1) + r"_delete");
                Exist = False
        log.info("creating %s \n" % (directory + r"/%s" % str(k+1)))
        os.mkdir(directory + r"/%s" % str(k+1))
		
    #with open('../data/TMU_all_training_data_s.json', 'r') as f:
        #tm_text = json.load(f)
		
    url = "http://dentaltw-info.uc.r.appspot.com/labels/completed"
    response = urllib.request.urlopen(url)
    tm_text = json.loads(response.read())
		
    #用kfold時的test index圖片名找到我們要predict的圖片
    for image_idx in test_index[k]:
        filename = str("https://storage.googleapis.com/taipei_medical/" + K_order[image_idx] + ".jpg")
		
        #記錄當下圖片的結果
        TTP=0
        TTN=0
        FFP=0
        FFN=0
		
        if int(K_order[image_idx]) == 267 or int(K_order[image_idx]) ==3726 or int(K_order[image_idx]) ==3725:
            continue
		
        skip = 0
        for key,value in tqdm(tm_text.items()):
            if skip:
                skip = 0
				
                continue
            raw_data = tm_text[key]

            url,filename2 = os.path.split(str(raw_data['filename']))
            filename2, suffix = os.path.splitext(filename2)
			
            if int(filename2) == 267 or int(filename2) ==3726 or int(filename2) ==3725:
                skip = 1#268 bug
				
            if(filename == raw_data['filename']):
                oimg = io.imread(filename)
                py, px = oimg.shape[:2]
		
                t_mask = []
                maxx = 0
                minx = px
                showmask = Image.new("L", [px,py], 0)
                caries = np.zeros((py,px),np.uint8)
                gums = np.zeros((py,px),np.uint8)
	
                for anno in raw_data['regions']:
                    if(anno == None):
                        continue
                    region = anno['region_attributes']

                    teeth = Image.new("L", [px,py], 0)
                    if region['type'] == 'tooth':
                        xs = anno['shape_attributes']['all_points_x']
                        ys = anno['shape_attributes']['all_points_y']
                        if(len(xs) < 3):
                            continue
                        maxx = max(max(xs),maxx)
                        minx = min(min(xs),minx)
                        combined = list(map(tuple, np.vstack((xs, ys)).T))
                        ImageDraw.Draw(teeth).polygon(combined, outline=1, fill=1)
                        teeth = np.array(teeth)
                        t_mask.append(teeth)
                        showmask = showmask + teeth
			
                    temp = Image.new("L", [px,py], 0)
                    if 'detail' in region and region['detail'] == 'caries':
                        xs = anno['shape_attributes']['all_points_x']
                        ys = anno['shape_attributes']['all_points_y']
                        combined = list(map(tuple, np.vstack((xs, ys)).T))

                        ImageDraw.Draw(temp).polygon(combined, outline=1, fill=1)
                        temp = np.array(temp)
                        temp = datahandlet.convert_bool_uint8(temp)
                        caries = caries + temp
		
        #讀取圖片
        overlay = oimg.copy()
        overlayo = oimg.copy()
		
        caries = datahandlet.convert_bool_uint8(caries) #將caries值map到0和255並轉成uint8
		
        t_mask = np.transpose((datahandlet.convert_bool_uint8(np.array(t_mask))), (1, 2, 0))
        twidth = int((maxx-minx)/(t_mask.shape[2]))#計算每顆牙齒的寬度

        edges = datahandlet.get_edges(t_mask, twidth) #利用縮小後的edge取框使得框在裏面一點
        
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
        oimg = oimg - (overlayo*0.3).astype(np.uint8) + (overlay *0.3).astype(np.uint8)
        nms_img = oimg.copy()
        nms_img2 = oimg.copy()
        
        #把框取出
        test_X = []
        test_Y = []
        predict_list = data_input.load_data("../data/cv/" + K_order[image_idx], \
                                        norm_size, class_num, predict = 1)
        if(len(predict_list) == 0):
            log.info("No data in %s \n" % ("../data/cv/" + K_order[image_idx]))
            continue
        
        sortlist_0 = []
        sortlist_1 = []

        plabel = []
        pscore = []
        
        #原本predict.py的部分
        for each in predict_list: #each is a tuple with (label,location,image)
            label,c,image = each[0],each[1],each[2]
            image = np.expand_dims(image, axis=0)#单张图片，改变维度
            result = model.predict(image)#分类预测
            proba = np.max(result)#最大概率
            predict_label = np.argmax(result)#提取最大概率下标
			
            if label == 1:
                if predict_label == 1:
                    TP += 1
                    TTP += 1
                    color = (int(colormap_true[0, int(np.ceil(255*1)), 0]), \
                            int(colormap_true[0, int(np.ceil(255*1)), 1]), \
                            int(colormap_true[0, int(np.ceil(255*1)), 2]))
                    cv2.rectangle(oimg,(c[1],c[0]),(c[3],c[2]),(color[0],color[1],color[2]),2)
                else:
                    FN += 1
                    FFN += 1
                    color = (int(colormap_true[0, int(np.ceil(255*0)), 0]), \
                            int(colormap_true[0, int(np.ceil(255*0)), 1]), \
                            int(colormap_true[0, int(np.ceil(255*0)), 2]))
                    cv2.rectangle(oimg,(c[1],c[0]),(c[3],c[2]),(color[0],color[1],color[2]),1)
            elif label == 0:
                if predict_label == 1:
                    FP += 1
                    FFP += 1
                    color = (int(colormap_true[0, int(np.ceil(255*1)), 0]), \
                            int(colormap_true[0, int(np.ceil(255*1)), 1]), \
                            int(colormap_true[0, int(np.ceil(255*1)), 2]))
                    cv2.rectangle(oimg,(c[1],c[0]),(c[3],c[2]),(color[0],color[1],color[2]),1)
                else:
                    TN += 1
                    TTN += 1
                    color = (int(colormap_true[0, int(np.ceil(255*0)), 0]), \
                            int(colormap_true[0, int(np.ceil(255*0)), 1]), \
                            int(colormap_true[0, int(np.ceil(255*0)), 2]))
                    cv2.rectangle(oimg,(c[1],c[0]),(c[3],c[2]),(color[0],color[1],color[2]),2)
            if predict_label == 0:
                sortlist_0.append((np.max(result),c,label))
            if predict_label == 1:
                sortlist_1.append((np.max(result),c,label))
                plabel.append(np.argmax(result))
                pscore.append(np.max(result))
            
            #顯示切出的單張框的結果
            '''
            plt.imshow(image[0],cmap='gray')#显示
            plt.title("label:{},predict_label:{}, proba:{:.2f}".format(label,predict_label,proba))
            #plt.savefig("../result/out/result.png")#.format(image[0]))
            plt.show()
            '''
        log.info("Doing NMS with image %s \n" % ("../data/cv/" + str(K_order[image_idx])))
        sortlist_0.sort(key=lambda x:x[0], reverse=True)
        sortlist_1.sort(key=lambda x:x[0], reverse=True)
        selectedist_0 = []
        selectedist_1 = []
        l_num = 0
        for NMS_data in [sortlist_0, sortlist_1]:
            selectedist = []
            sortlist = NMS_data.copy()
            while (sortlist): #NMS
                clearlist = []
                for num in range(1,len(sortlist)):
                    iou = datahandlet.intersection_over_union(sortlist[0][1], sortlist[num][1])
                    if iou > 0.1:
                        clearlist.append(num)
                while (clearlist):
                    del sortlist[clearlist[len(clearlist)-1]]
                    del clearlist[len(clearlist)-1]
                selectedist.append(sortlist[0])
                del sortlist[0]
            if l_num == 0: selectedist_0 = selectedist.copy();
            else: selectedist_1 = selectedist.copy();
            l_num = l_num + 1
        
        l_num = 0
		
        if plabel:
            pprecision, precall, pthresholds = precision_recall_curve(plabel, pscore)
            thresh = pthresholds[(int((len(pthresholds))*(0.45))) - 1]
        else:
            thresh = 0
		
        for NMS_result in [selectedist_0, selectedist_1]:
            for each in NMS_result: #each is a tuple with (proba,location,label)
                proba,c,label = each[0],each[1],each[2]

                
                if proba > 0:
                    distance = int((c[3]-c[1]) * 0.015)
                    size = ((c[3]-c[1]) * 0.008)

                    if label == 1:
                        if l_num == 1:
                            color = (int(colormap_true[0, int(np.ceil(255*1)), 0]), \
                                    int(colormap_true[0, int(np.ceil(255*1)), 1]), \
                                    int(colormap_true[0, int(np.ceil(255*1)), 2]))
                            cv2.rectangle(nms_img2,(c[1],c[0]),(c[3],c[2]),(color[0],color[1],color[2]),2)
                            cv2.putText(nms_img2, "%.2f%%" % (proba*100), (c[1], c[0]-distance), cv2.FONT_HERSHEY_SIMPLEX, size, (color[0],color[1],color[2]), 2)
                        else:
                            color = (int(colormap_true[0, int(np.ceil(255*1)), 0]), \
                                    int(colormap_true[0, int(np.ceil(255*1)), 1]), \
                                    int(colormap_true[0, int(np.ceil(255*1)), 2]))
                            #cv2.rectangle(nms_img2,(c[1],c[0]),(c[3],c[2]),(color[0],color[1],color[2]),1)
                            #cv2.putText(nms_img2, "-%.2f%%" % (proba*100), (c[1], c[0]-distance), cv2.FONT_HERSHEY_SIMPLEX, size, (color[0],color[1],color[2]), 1)
                    elif label == 0:
                        if l_num == 1:
                            color = (int(colormap_true[0, int(255*(0)), 0]), \
                                int(colormap_true[0, int(255*(0)), 1]), \
                                    int(colormap_true[0, int(255*(0)), 2]))
                            cv2.rectangle(nms_img2,(c[1],c[0]),(c[3],c[2]),(color[0],color[1],color[2]),2)
                            cv2.putText(nms_img2, "%.2f%%" % (proba*100), (c[1], c[0]-distance), cv2.FONT_HERSHEY_SIMPLEX, size, (255,100,0), 2)
                        else:
                            color = (int(colormap_true[0, int(255*(0)), 0]), \
                                    int(colormap_true[0, int(255*(0)), 1]), \
                                    int(colormap_true[0, int(255*(0)), 2]))
                            #cv2.rectangle(nms_img2,(c[1],c[0]),(c[3],c[2]),(color[0],color[1],color[2]),1)
                            #cv2.putText(nms_img2, "+%.2f%%" % (proba*100), (c[1], c[0]-distance), cv2.FONT_HERSHEY_SIMPLEX, size, (color[0],color[1],color[2]), 1)
                
				
                if proba > thresh:
                    distance = int((c[3]-c[1]) * 0.015)
                    size = ((c[3]-c[1]) * 0.008)

                    if label == 1:
                        if l_num == 1:
                            TPs += 1
                            color = (int(colormap_true[0, int(np.ceil(255*1)), 0]), \
                                    int(colormap_true[0, int(np.ceil(255*1)), 1]), \
                                    int(colormap_true[0, int(np.ceil(255*1)), 2]))
                            cv2.rectangle(nms_img,(c[1],c[0]),(c[3],c[2]),(color[0],color[1],color[2]),2)
                            cv2.putText(nms_img, "%.2f%%" % (proba*100), (c[1], c[0]-distance), cv2.FONT_HERSHEY_SIMPLEX, size, (color[0],color[1],color[2]), 2)
                        else:
                            FNs += 1
                            color = (int(colormap_true[0, int(np.ceil(255*1)), 0]), \
                                    int(colormap_true[0, int(np.ceil(255*1)), 1]), \
                                    int(colormap_true[0, int(np.ceil(255*1)), 2]))
                            #cv2.rectangle(nms_img,(c[1],c[0]),(c[3],c[2]),(color[0],color[1],color[2]),2)
                            #cv2.putText(nms_img, "-%.2f%%" % (proba*100), (c[1], c[0]-distance), cv2.FONT_HERSHEY_SIMPLEX, size, (color[0],color[1],color[2]), 1)
                    elif label == 0:
                        if l_num == 1:
                            FPs += 1
                            color = (int(colormap_true[0, int(255*(0)), 0]), \
                                int(colormap_true[0, int(255*(0)), 1]), \
                                    int(colormap_true[0, int(255*(0)), 2]))
                            cv2.rectangle(nms_img,(c[1],c[0]),(c[3],c[2]),(color[0],color[1],color[2]),2)
                            cv2.putText(nms_img, "%.2f%%" % (proba*100), (c[1], c[0]-distance), cv2.FONT_HERSHEY_SIMPLEX, size, (255,100,0), 2)
                        else:
                            TNs += 1
                            color = (int(colormap_true[0, int(255*(0)), 0]), \
                                    int(colormap_true[0, int(255*(0)), 1]), \
                                    int(colormap_true[0, int(255*(0)), 2]))
                           #cv2.rectangle(nms_img,(c[1],c[0]),(c[3],c[2]),(color[0],color[1],color[2]),2)
                            #cv2.putText(nms_img, "+%.2f%%" % (proba*100), (c[1], c[0]-distance), cv2.FONT_HERSHEY_SIMPLEX, size, (color[0],color[1],color[2]), 1)
            l_num = l_num + 1
        
        cv2.imwrite(outdir_nms+r"/%s/"%str(k+1) + "teeth ("+str(K_order[image_idx])+")nms.jpg", nms_img)
        cv2.imwrite(outdir_nms+r"/%s/"%str(k+1) + "teeth ("+str(K_order[image_idx])+")nms2.jpg", nms_img2)
        
        #產生所有牙齒的edge的圖
        t_edge = np.zeros((py,px),np.uint8)
        for loop in edges:
            t_edge = np.logical_or(t_edge,loop)
        t_edge = np.logical_not(t_edge)
        #將edge的部分的值變成0
        oimg [:,:,0] = oimg [:,:,0]*t_edge
        oimg [:,:,1] = oimg [:,:,1]*t_edge
        oimg [:,:,2] = oimg [:,:,2]*t_edge
        
        cv2.putText(oimg, "> Num "+str(TTP+TTN+FFP+FFN)+" -- TP: "+str(TTP)+" - TN: "+str(TTN)+" - FP: "+str(FFP)+" - FN: "+str(FFN), (int(py*0.2),int(py*0.9)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 1, cv2.LINE_AA)
        cv2.imwrite(outdir+r"/%s/"%str(k+1) + "teeth ("+str(image_idx)+").jpg", oimg)
        print(">"+str(image_idx)+"-----done")
    
    if (TP == 0) and (FP ==0):
        FPnz = 1
    else:
        FPnz = FP
		
    if (TPs == 0) and (FPs ==0):
        FPnzs = 1
    else:
        FPnzs = FPs
		
    print('----------------------------------------------------------------------------')
    print(f'	> Num {TP+TN+FP+FN} -- TP: {TP} - TN: {TN} - FP: {FP} - FN: {FN}	')
    print('----------------------------------------------------------------------------')
    print(" > PrecisionP: %.2f%% - PrecisionN: %.2f%% - Accuracy: %.2f%%"% ((TP/(TP+FPnz))*100, (TN/(TN+FN))*100, ((TN+TP)/(TP+TN+FP+FN))*100))
    print('----------------------------------------------------------------------------')
    print(f'	> Nms')
    print('----------------------------------------------------------------------------')
    print(f'	> Num {TPs+TNs+FPs+FNs} -- TP: {TPs} - TN: {TNs} - FP: {FPs} - FN: {FNs}	')
    print('----------------------------------------------------------------------------')
    print(" > PrecisionP: %.2f%% - PrecisionN: %.2f%% - Accuracy: %.2f%%"% ((TPs/(TPs+FPnzs))*100, (TNs/(TNs+FNs))*100, ((TNs+TPs)/(TPs+TNs+FPs+FNs))*100))
    print('----------------------------------------------------------------------------')
    cmat1 = confusion_matrix(TP,TN,FP,FN,modelnum=k,save_dir=outdir)
    cmat2 = confusion_matrix(TPs,TNs,FPs,FNs,modelnum=k,save_dir=outdir_nms)

        #plt.imshow(img)
        #plt.show()
        #plt.imshow(edges)
        #plt.savefig("../result/edge ("+str(num)+").jpg")
        #plt.show()