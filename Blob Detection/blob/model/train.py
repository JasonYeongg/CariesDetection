#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow import keras
import matplotlib.pylab as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend
import numpy as np
import cv2
import sys
sys.path.append("../process")#添加其他文件夹
import data_input #导入其他模块
from network import Lenet
from os import environ
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import math
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import shutil
from tqdm import tqdm

import logging
# log的部分
logging.basicConfig(level='INFO',
                    format='[%(asctime)s %(levelname)-8s %(module)-12s:Line %(lineno)-3d] %(message)s ',
                    datefmt='[%H:%M:%S]')

log = logging.getLogger('train.py')

def train(k,aug, model,train_x,train_y,test_x,test_y):
    log.info("Start training \n")
    initial_learning_rate = 0.0001#exponentially decay learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                  initial_learning_rate,decay_steps=80,decay_rate=0.98,staircase=True)#exponential_decay：decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
                  #400 0.98
                  #80 0.95
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule),metrics=['accuracy'])#配置
    #model.fit(train_x,train_y,batch_size,epochs,validation_data=(test_x,test_y))
    #add checkpoint to save best model
    #filepath="../checkpoint/weights-{epoch:02d}-{val_loss:.2f}.h5"
    maked = False
    while not os.path.exists("../checkpoint"):
        if not maked:
            log.warning("%s does not exist, creating...\n" % "../checkpoint")
            os.mkdir("../checkpoint"); maked = True
    filepath="../checkpoint/weights-best-%s-{epoch:02d}-{val_acc:.2f}.h5" %(k)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    class_weights = dict(zip(np.unique(train_y.argmax(1)),class_weight.compute_class_weight('balanced',np.unique(train_y.argmax(1)),train_y.argmax(1))))
    
    print("class weight before reweight: ",class_weights)
    if (class_weights[1]*0.8 > class_weights[0]):
        class_weights[1] = class_weights[1]*0.8
    print("class weight after reweight: ",class_weights)

    _history = model.fit_generator(aug.flow(train_x,train_y,batch_size=batch_size,seed=1),
                        validation_data=(test_x,test_y),steps_per_epoch=len(train_x)//batch_size,
                        epochs=epochs,callbacks=[checkpoint],verbose=1,class_weight=class_weights)
    
    model.save("../predict/model%s.h5" %(k))
    plt.style.use("ggplot")#matplotlib的美化样式
    plt.figure(1)
    N = epochs
    #model的history有四个属性，loss,val_loss,acc,val_acc
    plt.plot(np.arange(0,N),_history.history["loss"],label ="train_loss")
    plt.plot(np.arange(0,N),_history.history["val_loss"],label="val_loss")
    plt.plot(np.arange(0,N),_history.history["acc"],label="train_acc")
    plt.plot(np.arange(0,N),_history.history["val_acc"],label="val_acc")
    plt.title("loss and accuracy")
    plt.xlabel("epoch")
    plt.ylabel("loss/acc")
    plt.legend(loc="best")
    plt.ylim([0,3])
    plt.savefig("../result/result%s.png" %(k))
    plt.show(block=False)
    plt.pause(5)
    plt.close('all')
    return _history
	
def plot_labels(idx_of_caries,idx_of_normal,knum, save_dir=''):
    left = [1,2]
    height = [idx_of_caries,idx_of_normal]
    label = ['normal','caries']
    plt.bar(left, height, tick_label = label, width = 0.8, color = ['lightskyblue'])
    plt.ylabel('instances')

    plt.savefig(save_dir+"labels"+str(knum)+".png", dpi=250)
    plt.close()

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

if __name__ =="__main__":
    suppress_qt_warnings()
    channel = 1
    height = 64
    width = 64
    class_num = 2
    norm_size = 64#参数
    batch_size = int(input('Batch_size: '))
    epochs = int(input('Epochs: '))
    
    maked = False
    while not os.path.exists("../result"):
        if not maked:
            log.warning("%s does not exist, creating...\n" % "../result")
            os.mkdir("../result"); maked = True
    
    #Get image label(have caries not)
    images_X, images_Y = data_input.load_cvdata("../data/cv/")
    Kf_num = int(input('K fold number: '))
    kf = StratifiedKFold(n_splits = Kf_num, shuffle = True, random_state = 0)
    cvscores = []
    aug = ImageDataGenerator(
        rotation_range=15,
        #width_shift_range=0.1,
        #height_shift_range=0.1,
        #shear_range=0.2,
        #zoom_range=0.2,
        #samplewise_center=True,
        #samplewise_std_normalization=True,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest")#数据增强，生成迭代器
    
    #KFold Cross Validation
    k = 1
    hist = []
    C_matrix = []
    K_index = []
    for train_index,test_index in kf.split(images_X, images_Y.argmax(1)):
        K_index.append((train_index,test_index))
        train_X = []
        train_Y = []
        test_X = []
        test_Y = []
		
        tr = tqdm(train_index) # Create progress bar
        tr.set_description("loaded train_index")
        for idx in tr:
            tmp_X,tmp_Y = data_input.load_data("../data/cv/" + images_X[idx], \
                                        norm_size, class_num)
            if(len(tmp_X) != 0):
                train_X.append(tmp_X)
                train_Y.append(tmp_Y)
        train_X = np.concatenate(train_X); train_Y = np.concatenate(train_Y);
		
        te = tqdm(test_index) # Create progress bar
        te.set_description("loaded test_index")
        for idx in te:
            tmp_X,tmp_Y = data_input.load_data("../data/cv/" + images_X[idx], \
                                        norm_size, class_num)
            if(len(tmp_X) != 0):
                test_X.append(tmp_X)
                test_Y.append(tmp_Y)
        test_X = np.concatenate(test_X); test_Y = np.concatenate(test_Y);
        
        #複製caries資料使得class的數量一樣，所以可以去掉class weight---------------------------------------------------
        oversampling = 0
        idx_of_caries = np.where(train_Y.argmax(axis=1) == 1)
        idx_of_normal = np.where(train_Y.argmax(axis=1) == 0)
		
        print("Training : caries = ", len(idx_of_caries[0]), "  normal = ", len(idx_of_normal[0]))
		
        plot_labels(int(len(idx_of_caries[0])),int(len(idx_of_normal[0])),knum=k,save_dir="../result/")
		
        distance = int(len(idx_of_normal[0]) - len(idx_of_caries[0]))
        pbar = tqdm(total=distance)
        pbar.set_description("oversampling train_index")
        for ovrs in range(distance):
            train_X = np.append(train_X, train_X[idx_of_caries[0][oversampling],:,:].reshape(1,norm_size,norm_size,1),axis=0)
            train_Y = np.append(train_Y,train_Y[idx_of_caries[0][oversampling],:].reshape(1,2),axis=0)
            
            oversampling = oversampling + 1
            idx_of_caries = np.where(train_Y.argmax(axis=1) == 1)
            idx_of_normal = np.where(train_Y.argmax(axis=1) == 0)
            pbar.update(1)
        print("Training : caries = ", len(idx_of_caries[0]), "  normal = ", len(idx_of_normal[0]))
        log.info("Oversampling with %d data \n" % oversampling)
        
        oversampling = 0
        idx_of_caries = np.where(test_Y.argmax(axis=1) == 1)
        idx_of_normal = np.where(test_Y.argmax(axis=1) == 0)
        print("Validation : caries = ", len(idx_of_caries[0]), "  normal = ", len(idx_of_normal[0]))
		
        distance = int(len(idx_of_normal[0]) - len(idx_of_caries[0]))
        pbar2 = tqdm(total=distance)
        pbar2.set_description("oversampling test_index")
        for ovrs in range(distance):
            test_X = np.append(test_X, test_X[idx_of_caries[0][oversampling],:,:].reshape(1,norm_size,norm_size,1),axis=0)
            test_Y = np.append(test_Y,test_Y[idx_of_caries[0][oversampling],:].reshape(1,2),axis=0)
            
            oversampling = oversampling + 1
            idx_of_caries = np.where(test_Y.argmax(axis=1) == 1)
            idx_of_normal = np.where(test_Y.argmax(axis=1) == 0)
            pbar2.update(1)
        print("Validation : caries = ", len(idx_of_caries[0]), "  normal = ", len(idx_of_normal[0]))
        log.info("Oversampling with %d data \n" % oversampling)
        #-------------------------------------------------------------------------------------------------------------
        
        log.info("Training with %d data \n" % train_X.shape[0])
        log.info("Validation with %d data \n" % test_X.shape[0])
        
        model = Lenet.neural(channel=channel, height=height,width=width, classes=class_num)#网络
        hist.append(train(k,aug,model,train_X,train_Y,test_X,test_Y))#训练
        scores = model.evaluate(test_X,test_Y,verbose=0)
        Y_pred = model.predict(test_X)
        C_matrix.append(confusion_matrix(test_Y.argmax(axis=1), Y_pred.argmax(axis=1)))
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        backend.clear_session()#清除backend
        k = k+1
        #break #跑一次就結束
    
    #show confution matrix
    for c in C_matrix:
        print(c)
    #Save batch size,epochs and Kfold index
    fw = open('K_index.txt','w')
    fw.write("batch size: " + str(batch_size) + "\n")
    fw.write("epochs: " + str(epochs) + "\n")
    fw.write(str(Kf_num) + "\n")
    fw.write('\n')
    for tpl in K_index:
        for idx in tpl:
            for idx2,num in enumerate(idx):
                fw.write(str(num)) if idx2==0 else fw.write("," + str(num))
            fw.write('\n')
        fw.write('\n')
    fw.write('Order:\n')
    for idx in range(len(images_X)):
        fw.write(images_X[idx])
        fw.write('\n')
    fw.close()
    
    #plot graph together
    plt.figure(2)
    plt.style.use("ggplot")#matplotlib的美化样式
    N = epochs
    #model的history有四个属性，loss,val_loss,acc,val_acc
    fig,ax = plt.subplots(3,3,figsize=(12,8))
    fig.text(0.5,0.04,"epoch",ha='center')
    fig.text(0.04,0.5,"loss/accuracy",va='center',rotation='vertical')
    for i in range(k-1):
        ax[math.floor((i)/3),(i)%3].plot(np.arange(0,N),hist[i].history["loss"],label ="train_loss")
        ax[math.floor((i)/3),(i)%3].plot(np.arange(0,N),hist[i].history["val_loss"],label="val_loss")
        ax[math.floor((i)/3),(i)%3].plot(np.arange(0,N),hist[i].history["acc"],label="train_acc")
        ax[math.floor((i)/3),(i)%3].plot(np.arange(0,N),hist[i].history["val_acc"],label="val_acc")
        ax[math.floor((i)/3),(i)%3].set_title('result%s' %(i+1))
        ax[math.floor((i)/3),(i)%3].legend(loc="best")
        ax[math.floor((i)/3),(i)%3].set_ylim([0,3])
    plt.suptitle("cv scores = %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    fig.savefig("../result/cv_result.png") 
    plt.show(block=False)
    plt.pause(5)
    plt.close('all')
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
