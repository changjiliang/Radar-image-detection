import os
from PIL import Image
import PIL.Image
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Conv2D, MaxPooling2D,normalization
from keras.callbacks import TensorBoard
import math
import keras
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from keras import regularizers
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf



#from keras.layers import LeakyReLU
# -----------------------------------------------------------------------------------------
# 将训练集图片转换成数组
ima1 = os.listdir('./train_5400')

def read_image1(filename):
    img = Image.open('./train_5400/'+filename).convert('RGB')
    return np.array(img)

x_train = []

for i in ima1:
    x_train.append(read_image1(i))

x_train = np.array(x_train)

# 根据文件名提取标签
y_train = []

for filename in ima1:
    y_train.append(int(filename.split('.')[0]))

y_train = np.array(y_train)

# -----------------------------------------------------------------------------------------
# 将测试集图片转化成数组
ima2 = os.listdir('./test_600')

def read_image2(filename):
    img = Image.open('./test_600/'+filename).convert('RGB')
    return np.array(img)

x_test = []

for i in ima2:
    x_test.append(read_image2(i))

x_test = np.array(x_test)

# 根据文件名提取标签
y_test = []
for filename in ima2:
    y_test.append(int(filename.split('.')[0]))

y_test = np.array(y_test)

#-------------------------------------------------------------------------------------
# 将标签转换格式
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 将特征点从0~255转换成0~1提高特征提取精度
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# 搭建卷积神经网络# 搭建卷积神经网络# 搭建卷积神经网络# 搭建卷积神经网络



#layer1
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),
                 padding='same',activation='relu',strides=(1,1),input_shape=(32,32,3)))#,kernel_regularizer=regularizers.l2(0.01)))
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same',strides=(2,2)))
model.add(normalization.BatchNormalization(epsilon=1e-06,mode=0,axis=-1,momentum=0.9,weights=None,beta_initializer='zero',gamma_initializer='one'))
#model.add(Dropout(0.5))
#layer2
model.add(Conv2D(filters=64,kernel_size=(3,3),
                 padding='same',activation='relu',strides=(1,1)))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same',strides=(2,2)))
model.add(normalization.BatchNormalization(epsilon=1e-06,mode=0,axis=-1,momentum=0.9,weights=None,beta_initializer='zero',gamma_initializer='one'))
#model.add(Dropout(0.5))
#layer3
model.add(Conv2D(filters=128,kernel_size=(3,3),
                 padding='same',activation='relu',strides=(1,1)))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same',strides=(2,2)))
model.add(normalization.BatchNormalization(epsilon=1e-06,mode=0,axis=-1,momentum=0.9,weights=None,beta_initializer='zero',gamma_initializer='one'))
#model.add(Dropout(0.5))

#layer4
model.add(Conv2D(filters=256,kernel_size=(3,3),
                 padding='same',activation='relu',strides=(1,1)))
#model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same',strides=(2,2)))
model.add(normalization.BatchNormalization(epsilon=1e-06,mode=0,axis=-1,momentum=0.9,weights=None,beta_initializer='zero',gamma_initializer='one'))
#model.add(Dropout(0.5))
# #layer5
model.add(Conv2D(filters=512,kernel_size=(3,3),
                 padding='same',activation='relu',strides=(1,1)))
model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same',strides=(2,2)))
#model.add(normalization.BatchNormalization(epsilon=1e-06,mode=0,axis=-1,momentum=0.9,weights=None,beta_initializer='zero',gamma_initializer='one'))
#model.add(Dropout(0.6))

#全连接
model.add(Flatten())
model.add(Dense(2048,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

model.summary()
############# 动态学习率
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.6
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate


lrate = keras.callbacks.LearningRateScheduler(step_decay)
sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=True)


#sgd = SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

history = model.fit(x_train,
          y_train,
          batch_size=100,
          epochs=32,
          shuffle=True,
          validation_data=(x_test,y_test),
          callbacks=[lrate])
model.save_weights('ship_weights.h5', overwrite=True)

score = model.evaluate(x_test, y_test, batch_size=50)
print(score)


# summarize history for accuracy
plt.plot(history.history['acc'],'--+')
plt.plot(history.history['val_acc'],'--+')

plt.title('model-accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.ylim(0,2)
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
my_x_ticks = np.arange(0,30,1)
plt.xticks(my_x_ticks)
plt.xticks([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
           ['1','3','5','7','9','11','13','15','17','19','21',
            '23','25','27','29','31'])
# plt.xticks([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,
#             54,56,58,60,62,64,66,68,70,72,74,76,78,80],
#            ['1','3','5','7','9','11','13','15','17','19','21',
#             '23','25','27','29','31','33','35','37','39','41','43','45','47','49','51','53','55','57','59','61',
#             '63','65','67','69','71','73','75','77','79'])
plt.savefig('./accuracyVSepoch.png')
plt.show()


# summarize history for loss
plt.plot(history.history['loss'],'--+')
plt.plot(history.history['val_loss'],'--+')
plt.title('model-loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim(0,1)
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
my_x_ticks = np.arange(0,30,1)
xmajorLocator = MultipleLocator(2)
plt.xticks(my_x_ticks)
plt.xticks([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
           ['1','3','5','7','9','11','13','15','17','19','21',
            '23','25','27','29','31'])
plt.savefig('./lossVSepoch.png')
plt.show()
