from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.models import Model
from matplotlib import pyplot as plt
import cv2


def label2onehot(labels):
    new_lbs = []
    for i in labels:
        t_lb = [0]*10
        t_lb[i] = 1
        new_lbs.append(t_lb)
    return np.array(new_lbs).reshape(labels.shape[0], 10)

mnist = input_data.read_data_sets("../../LabPractice/MNIST_data", reshape=False, one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels


x_train = x_train.reshape(x_train.shape[0], 28, 28)
# y_train = label2onehot(y_train)
x_test = x_test.reshape(x_test.shape[0], 28, 28)
# y_test = label2onehot(y_test)

def data2img(data):
    img = (data*255).astype(np.uint8)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

# img = (x_train[0]*255).astype(np.uint8)
# img = cv2.resize(img, (200, 200))
# # print(img, type(img))
# cv2.imshow("233", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 特征提取器, 得到fc2层输出
model = keras.applications.VGG16(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

import time
tic = time.clock()

features = []
x_train = x_train[:1000]
for i, x in enumerate(x_train):
    if i%100 == 0:
        toc = time.clock()
        elap = toc-tic
        print("analyzing image %d / %d. Time: %4.4f seconds." % (i, len(x_train),elap))
        tic = time.clock()
    x = data2img(x)
    feat = feat_extractor.predict(x.reshape(-1, 224, 224, 3))
    features.append(feat)

# 降维
from sklearn.decomposition import PCA
features = np.array(features)
pca = PCA(n_components=300)
pca.fit(features)
pca_features = pca.transform(features)