
# coding: utf-8

# In[1]:

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm_notebook
from random import shuffle
import shutil
import pandas as pd


# In[2]:

# 重新构建文件树
def organize_datasets(path_to_data, n=4000, ratio=0.2):
    # 读取文件目录中的内容
    files = os.listdir(path_to_data)
    files = [os.path.join(path_to_data, f) for f in files]
    shuffle(files)
    files = files[:n]

    n = int(len(files) * ratio)
    # 分割25000数据集， 80%作为训练集， 20%作为测试集
    val, train = files[:n], files[n:]

    # 删除文件目录
    shutil.rmtree('./data/')
    print('/data/ removed')

    # 建立文件目录
    for c in ['dogs', 'cats']:
        os.makedirs('./data/train/{0}/'.format(c))
        os.makedirs('./data/validation/{0}/'.format(c))

    print('folders created !')

    # 用进度条展示data copy的进度
    for t in tqdm_notebook(train):
        if 'cat' in t:
            shutil.copy2(t, os.path.join('.', 'data', 'train', 'cats'))
        else:
            shutil.copy2(t, os.path.join('.', 'data', 'train', 'dogs'))

    for v in tqdm_notebook(val):
        if 'cat' in v:
            shutil.copy2(v, os.path.join('.', 'data', 'validation', 'cats'))
        else:
            shutil.copy2(v, os.path.join('.', 'data', 'validation', 'dogs'))

    print('Data copied!')

# 设置参数
ratio = 0.2
n = 25000
# 从以下目录重新建立文件，用作训练和测试
organize_datasets(path_to_data='D:\\1python notes\\cats vs dogs\\train', n=n, ratio=ratio)


# In[3]:

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras_tqdm import TQDMNotebookCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback


# In[4]:

'''
创建两个 ImageDataGenerator 对象。
train_datagen 对应训练集，
val_datagen 对应测试集，两者都会对图像进行缩放
'''
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1 / 255.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True
                                   )
val_datagen = ImageDataGenerator(rescale=1 / 255.)


# In[5]:

# 创建 train_generator and validation_generator
train_generator = train_datagen.flow_from_directory(
    './data/train/',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    './data/validation/',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical')


# In[10]:

'''
使用 fit_generator 方法，它是一个将生成器作为输入的变体（标准拟合方法）。
我们训练模型的时间超过 50 个 epoch。
'''
fitted_model = model.fit_generator(
    train_generator,
    steps_per_epoch=int(n * (1 - ratio)) // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=int(n * ratio) // batch_size,
    verbose=0)


# In[ ]:




# In[ ]:




# In[ ]:

'''
用VGG16，InceptionV3, ResNet50做迁移学习，
这些都是ImageNet上面预训练过的网络，
加载网络的权重到所有的卷积层。
它将作为特征提取器来提取图像特征，在连接到全连接层
'''

from keras import applications

# 下载并加载网络和参数，不包括网络顶部的全连接层，这个我们之后根据需要自己定义
# model = applications.VGG16(include_top=False, weights='imagenet')
model = applications.InceptionV3(include_top=False, weights='imagenet')
# model = applications.ResNet50(include_top=False, weights='imagenet')
datagen = ImageDataGenerator(rescale=1. / 255)


# In[ ]:

'''
将图像传进网络提取特征表示，然后连接到全连接网络来分类
'''
generator = datagen.flow_from_directory('./data/train/',
                                        target_size=(150, 150),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)

# 生成并保存训练数据的特征
bottleneck_features_train = model.predict_generator(generator, int(n * (1 - ratio)) // batch_size)
np.save(open('D:\\1python notes\\cats vs dogs\\features\\bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


# In[ ]:

generator = datagen.flow_from_directory('./data/validation/',
                                        target_size=(150, 150),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)

# 生成并保存测试数据的特征
bottleneck_features_validation = model.predict_generator(generator, int(n * ratio) // batch_size)
np.save('D:\\1python notes\\cats vs dogs\\features\\bottleneck_features_validation.npy', bottleneck_features_validation)


# In[ ]:

'''
为每张传入图片关联上标签
'''
train_data = np.load('D:\\1python notes\\cats vs dogs\\features\\bottleneck_features_train.npy')
train_labels = np.array([0] * (int((1-ratio) * n) // 2) + [1] * (int((1 - ratio) * n) // 2))

validation_data = np.load('D:\\1python notes\\cats vs dogs\\features\\bottleneck_features_validation.npy')
validation_labels = np.array([0] * (int(ratio * n) // 2) + [1] * (int(ratio * n) // 2))


# In[ ]:

'''
自己构造一个全连接网络，
附加上从 VGG16，InceptionV3, ResNet50 中抽取到的特征，
我们将它作为 CNN 的分类部分
'''
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(1024, activation='relu'))
# 添加dropout防止过拟合
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 定义优化器
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])



# In[ ]:

# 训练网络
fitted_model = model.fit(train_data, train_labels,
                         epochs=15,
                         batch_size=batch_size,
                         validation_data=(validation_data, validation_labels[:validation_data.shape[0]]),
                         verbose=0,
                         )


# In[ ]:

'''
在 15 个 epoch 后，
VGG16 模型达到了 90.9% 的准确度
InceptionV3 模型达到了95.4% 的准确度

'''


# In[ ]:




# In[ ]:



