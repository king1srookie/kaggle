
# coding: utf-8

# **Humpback Whale Identification - CNN with Keras**
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
import zipfile as zf
import os
import csv
import gc
import operator
import random
from sklearn.model_selection import train_test_split
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from random import shuffle
from IPython.display import Image
from pathlib import Path

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import np_utils
import keras.backend as K
from keras.models import Sequential
from keras import optimizers


trainData = pd.read_csv("../input/whale-categorization-playground/train.csv")#dataframe格式


trainData.sample(5)#随机选取值





Image(filename="../input/whale-categorization-playground/train/"+random.choice(trainData['Image'])) 


def prepareImages(data, m, dataset):#准备训练数据，改变形状并转换成数组  X = prepareImages(trainData, 9850, "train")
    
    print("Preparing images")
    
    X_train = np.zeros((m, 100, 100, 3))
    
    count = 0
    
    for fig in data['Image']:
        img = image.load_img("../input/whale-categorization-playground/"+dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)#对样本执行 逐样本均值消减 的归一化，即在每个维度上减去样本的均值
        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    count = 0
    
    print("Finished!")
            
    return X_train
def prepareY(Y):#准备标签并变为one-hot编码 Y = trainData['Id']，就是属于的是哪条

    values = array(Y)#numpy.array创建数组
    print('训练集标签的形状')
    print(values.shape)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)#就知道有多少个不同的属性值，有male和female，就用0和1表示，
    #le = LabelEncoder() le.fit([1,5,67,100]) le.transform([1,1,100,67,5])  输出： array([0,0,3,2,1])
    # 假如有3个不同的值，就用0,1,2表示。step2中transform操作就是转为数字表示形式。
    print('label编码')
    print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)#ohe.fit([[1],[2],[3],[4]]) ohe.transform([2],[3],[1],[4]).toarray()
    # 输出：[ [0,1,0,0] , [0,0,1,0] , [1,0,0,0] ,[0,0,0,1] ]
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)

    y = onehot_encoded
    print('热编码大小')
    print(y.shape)#(9850, 4251) 因为总共有4251类
    return y, label_encoder

#建立网络
mod = Sequential()

mod.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))

mod.add(BatchNormalization(axis = 3, name = 'bn0'))#该层在每个batch上将前一层的激活值重新规范化，
# 即使得其输出数据的均值接近0，其标准差接近1  axis: 整数，指定要规范化的轴，通常为特征轴
mod.add(Activation('relu'))

mod.add(MaxPooling2D((2, 2), name='max_pool'))
mod.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
mod.add(Activation('relu'))
mod.add(AveragePooling2D((3, 3), name='avg_pool'))

mod.add(Flatten())
mod.add(Dense(500, activation="relu", name='rl'))
mod.add(Dropout(0.8))
mod.add(Dense(4251, activation='softmax', name='sm'))
print('输出维数')
print(mod.output_shape)

mod.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


X = prepareImages(trainData, 9850, "train")

X /= 255

print("Shape X-train: ", X.shape)


Y = trainData['Id']

print("Shape Y-train: ", Y.shape)


y, label_encoder = prepareY(Y)#y是热编码


history = mod.fit(X, y, epochs=100, batch_size=100, verbose=1)
gc.collect()

plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()


test = os.listdir("../input/whale-categorization-playground/test/")# 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
print(len(test))


col = ['Image']
testData1 = pd.DataFrame(test[0:3899], columns=col)
testData2 = pd.DataFrame(test[3900:7799], columns=col)
testData3 = pd.DataFrame(test[7800:11699], columns=col)
testData4 = pd.DataFrame(test[11700:15609], columns=col)
testData = pd.DataFrame(test, columns=col)


gc.collect()
X = prepareImages(testData1, 3900, "test")
X /= 255

predictions1 = mod.predict(np.array(X), verbose=1)
gc.collect()


X = prepareImages(testData2, 3900, "test")
X /= 255
predictions2 = mod.predict(np.array(X), verbose=1)
gc.collect()

X = prepareImages(testData3, 3900, "test")
X /= 255
predictions3 = mod.predict(np.array(X), verbose=1)
gc.collect()

X = prepareImages(testData4, 3910, "test")
X /= 255
predictions4 = mod.predict(np.array(X), verbose=1)
gc.collect()

predictions = np.concatenate((predictions1, predictions2), axis=0)
predictions = np.concatenate((predictions, predictions3), axis=0)
predictions = np.concatenate((predictions, predictions4), axis=0)#拼接
gc.collect()
print(predictions.shape)
print(predictions)

print(predictions.shape)

copy_pred = np.copy(predictions)#若对初始变量进行改变，普通的等号会让关联的变量发生相同的改变（以前竟然没有注意到Python的这个特性）
print('copy_pred')
print(copy_pred)
# ，np.copy()的变量则不会改变
idx = np.argmax(copy_pred, axis=1)#输出每行中数值最大的索引
print('idx')
print(idx)
copy_pred[:,idx] = 0#这些列全部被换成0,,这里好像有问题，每张图的结果会相互干扰。
idx2 = np.argmax(copy_pred, axis=1)
print('idx2')
print(idx2)
copy_pred[:, idx2] = 0
idx3 = np.argmax(copy_pred, axis=1)
copy_pred[:, idx3] = 0
idx4 = np.argmax(copy_pred, axis=1)
copy_pred[:, idx4] = 0
idx5 = np.argmax(copy_pred, axis=1)


results = []

print(idx[0:10])#[  0   0   0   0   0   0 999   0   0   0]
print(idx2[0:10])#[3794  189 2351 3651 2351 2554  483 1991 1048 4236]
print(idx3[0:10])#[3221 3688 3569 3812 1501 1891 4097 1691  772  627]
print(idx4[0:10])#[3466 3935   77  235 2534 2615 1295  893 3135 3285]
print(idx5[0:10])#[0 0 0 0 0 0 0 0 0 0]
threshold = 0.05 #threshold - only consider answers with a probability higher than it
print(predictions.shape[0])
for i in range(0, predictions.shape[0]):#15610
    each = np.zeros((4251, 1))
    each2 = np.zeros((4251, 1))
    each3 = np.zeros((4251, 1))
    each4 = np.zeros((4251, 1))
    each5 = np.zeros((4251, 1))
    if((predictions[i, idx5[i]] > threshold)):
        each5[idx5[i]] = 1#第五大的坐标
        each4[idx4[i]] = 1
        each3[idx3[i]] = 1
        each2[idx2[i]] = 1
        each[idx[i]] = 1
        tags = [label_encoder.inverse_transform([argmax(each)])[0], label_encoder.inverse_transform([argmax(each2)])[0], label_encoder.inverse_transform([argmax(each3)])[0], label_encoder.inverse_transform([argmax(each4)])[0], label_encoder.inverse_transform([argmax(each5)])[0]]
    else:
        if((predictions[i, idx4[i]] > threshold)):
            print(predictions[i, idx4[i]])
            each4[idx4[i]] = 1
            each3[idx3[i]] = 1
            each2[idx2[i]] = 1
            each[idx[i]] = 1#[0]是取出里面的字符串
            tags = [label_encoder.inverse_transform([argmax(each)])[0], label_encoder.inverse_transform([argmax(each2)])[0], label_encoder.inverse_transform([argmax(each3)])[0], label_encoder.inverse_transform([argmax(each4)])[0]]
        else:
            if((predictions[i, idx3[i]] > threshold)):
                each3[idx3[i]] = 1
                each2[idx2[i]] = 1
                each[idx[i]] = 1
                tags = [label_encoder.inverse_transform([argmax(each)])[0], label_encoder.inverse_transform([argmax(each2)])[0], label_encoder.inverse_transform([argmax(each3)])[0]]
            else:
                if((predictions[i, idx2[i]] > threshold)):
                    each2[idx2[i]] = 1
                    each[idx[i]] = 1
                    tags = [label_encoder.inverse_transform([argmax(each)])[0], label_encoder.inverse_transform([argmax(each2)])[0]]
                else:
                    each[idx[i]] = 1
                    tags = label_encoder.inverse_transform([argmax(each)])[0]
    results.append(tags)


myfile = open('output.csv','w')

column= ['Image', 'Id']

wrtr = csv.writer(myfile, delimiter=',')
wrtr.writerow(column)

for i in range(0, testData.shape[0]):
    pred = ""
    if(len(results[i])==5):
        if (results[i][4]!=results[i][0]):
            pred = results[i][0] + " " + results[i][1] + " " + results[i][2] + " " + results[i][3] + " " + results[i][4]
        else:
            pred = results[i][0] + " " + results[i][1] + " " + results[i][2] + " " + results[i][3]
    else:
        if(len(results[i])==4):
            pred = results[i][0] + " " + results[i][1] + " " + results[i][2] + " " + results[i][3]
        else:
            if(len(results[i])==3):
                pred = results[i][0] + " " + results[i][1] + " " + results[i][2]
            else:
                if(len(results[i])==2):
                    pred = results[i][0] + " " + results[i][1]
                else:
                    pred = results[i]
            
    result = [testData['Image'][i], pred]
    wrtr.writerow(result)
    
myfile.close()





