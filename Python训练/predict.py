import os
from PIL import Image
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Conv2D, MaxPooling2D

def prepicture(picname):
    img = Image.open('./prediction/' + picname)
    new_img = img.resize((28, 28), Image.BILINEAR)
    new_img.save(os.path.join('./prediction/', os.path.basename(picname)))

def read_image2(filename):
    img = Image.open('./prediction/' + filename).convert('RGB')
    return np.array(img)


prepicture('1.1951.png')
x_test = []

x_test.append(read_image2('1.1951.png'))

x_test = np.array(x_test)

x_test = x_test.astype('float32')
x_test /= 255


#layer1
model = Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),
                 padding='same',activation='relu',strides=(1,1),input_shape=(28,28,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same',strides=(2,2)))
model.add(Dropout(0.25))
#layer2
model.add(Conv2D(filters=256,kernel_size=(3,3),
                 padding='same',activation='relu',strides=(1,1)))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same',strides=(2,2)))
model.add(Dropout(0.25))
#全连接
model.add(Flatten())
model.add(Dense(2048,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.load_weights('ship_weights.h5')

classes = model.predict_classes(x_test)[0]
print(classes)