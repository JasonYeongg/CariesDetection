#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout,SpatialDropout2D, Dense, Flatten,GlobalMaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

class Lenet:

    def neural(channel,height,width,classes):
        input_shape = (channel,height,width)
        if K.image_data_format() == "channels_last":#确认输入维度
            input_shape = (height,width,channel)
        model = Sequential()#顺序模型（keras中包括顺序模型和函数式API两种方式）
        
        model.add(Conv2D(32,(3,3),padding="same",activation="relu",input_shape=input_shape,name="conv1"))
        model.add(Dropout(0.3))#0.3
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name="pool1"))
        
        model.add(Conv2D(64,(3,3),padding="same",activation="relu",name="conv2",))
        model.add(Dropout(0.3))#0.3
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name="pool2"))
        
        model.add(Conv2D(128,(3,3),padding="same",activation="relu",name="conv3",))
        model.add(Dropout(0.5))#0.5
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name="pool3"))
        
        model.add(Flatten())
        model.add(Dropout(0.3))#0.25
        
        model.add(Dense(128,kernel_initializer=keras.initializers.Zeros(),bias_initializer=keras.initializers.Ones(),kernel_regularizer=l2(0.05), bias_regularizer=l2(0.01),activation="relu",name="fc1"))
        model.add(Dropout(0.3))#0.3
        
        model.add(Dense(256,kernel_initializer=keras.initializers.Zeros(),bias_initializer=keras.initializers.Ones(),kernel_regularizer=l2(0.05), bias_regularizer=l2(0.01),activation="relu",name="fc2"))
        model.add(Dropout(0.5))#0.4
        
        model.add(Dense(classes,activation="softmax",name="fc3"))
        # 看每一层layer有多少个被训练的weights和bias（kernel指weights）
        #model.trainable_weights
        model.summary()
        return model