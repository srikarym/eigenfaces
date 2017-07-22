
# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold
from sklearn.decomposition import FastICA

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# In[2]:

# print X_train.shape,h,w
# lyt = X[0]
# lyt = lyt.reshape((h,w))
# plt.imshow(lyt,'gray')
# print X_train.shape
import cv2
from keras.utils import np_utils

print h,w
X_train_reshaped = np.zeros((X_train.shape[0],1,32,32))
for i in xrange(X_train.shape[0]):
    X_train_reshaped[i,:,:] = cv2.resize(X_train[i].reshape((h,w)),(32,32))
print X_train_reshaped.shape

X_test_reshaped = np.zeros((X_test.shape[0],1,32,32))
for i in xrange(X_test.shape[0]):
    X_test_reshaped[i,:,:] = cv2.resize(X_test[i].reshape((h,w)),(32,32))
print X_test_reshaped.shape
#plt.imshow(X_test_reshaped[0,0])

Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)


# In[3]:

count = np.zeros(7)
for i in xrange(len(y)):
    count[y[i]] = count[y[i]] + 1
print count
class_weight = {0 : 1.0/count[0],
        1: 1.0/count[1],
        2: 1.0/count[2],
        3: 1.0/count[3],
        4: 1.0/count[4],
        5: 1.0/count[5],
        6: 1.0/count[6]}


# In[ ]:

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras import backend as K
K.set_image_dim_ordering('th')
import json

model = keras.models.Sequential()
model.add(keras.layers.convolutional.Convolution2D(8,3,3,input_shape=(1, 32, 32),border_mode='same',activation='relu',W_constraint=maxnorm(3)))
model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.35))

#model.add(keras.layers.convolutional.Convolution2D(8,3,3,activation='relu',border_mode='same',W_constraint=maxnorm(3)))
#model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(40,activation='relu',W_constraint=maxnorm(3)))
model.add(keras.layers.Dense(7,activation='softmax'))

json_txt = model.to_json()
# print json_txt
with open('model4.json','w') as outfile:
    json.dump(json_txt,outfile)
outfile.close()
epochs = 50
# opt = SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)
# opt = SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=True)
# opt = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=opt,class_weight = class_weight, metrics=['accuracy'])

filepath="model_weights4.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# model.load_weights("model_weights4.hdf5")
# opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.fit(X_train_reshaped, Y_train,validation_data=(X_test_reshaped,Y_test),callbacks=callbacks_list, nb_epoch=200, batch_size=32,verbose = 1)


# In[ ]:

model.load_weights(filepath)
(loss, accuracy) = model.evaluate(X_test_reshaped,Y_test,verbose=0)
print 'test accuracy : ',accuracy

