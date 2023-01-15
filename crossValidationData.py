from keras.utils import Sequence
from keras.utils import to_categorical
import numpy as np

from dataset_utils import *

from augmentation import *


import cv2
from sklearn.model_selection import train_test_split

import os
os.environ['PYTHONHASHSEED'] = '0'

import random
from random import seed, randint
seed(1234)

import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.set_random_seed(4321)



# Normalize a dataset using min-max normalization (let the values between 0 an 1)
def normalizeDataset(x, minValue=None, maxValue=None):
    if minValue is None: minValue = np.min(x)
    if maxValue is None: maxValue = np.max(x)

    print('MIN_MAX: ', minValue, maxValue)

    return (x - minValue)/(maxValue - minValue)

def calculateMeanDataset(x):
    soma = [0] * 3
    soma[0] = 0
    soma[1] = 0
    soma[2] = 0
    count = 0

    for img in x:
        media = np.mean(img, axis = (0,1))
        count = count + 1
        for i in range(0,3):
            soma[i] = soma[i] + media[i]

    meanTotal = [0] * 3
    for i in range(0,3):
        meanTotal[i] = soma[i]/count

    return meanTotal

def extractMean(x, mean, std=None):
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]

    return x

def preprocess(x, meanTotal=None, minValue=None, maxValue=None):

    if meanTotal is None: meanTotal = calculateMeanDataset(x)
    x = extractMean(x, meanTotal)

    if minValue is None: minValue = np.min(x)
    if maxValue is None: maxValue = np.max(x)

    minValue = float(minValue)
    maxValue = float(maxValue)

    x = normalizeDataset(x, minValue, maxValue)


    print('\n==============', meanTotal, minValue, maxValue, '\n')

    return x, meanTotal, minValue, maxValue

def saveMeanMinMax(meanTotal, minValue, maxValue, save_dir = './', name = 'mean-min-max'):
    file = open(save_dir + name + ".txt", "w")
    [ file.write(str(v) + '\n') for v in meanTotal ]
    file.write(str(minValue) + '\n')
    file.write(str(maxValue))
    file.close()


def saveCrossValidData(x, y, paths, name='', meanTotal=None, minValue=None, maxValue=None):
    savePath = './CrossValidData/'
    if not os.path.exists(savePath): os.makedirs(savePath)

    data, mean, minVal, maxVal = preprocess(x, meanTotal, minValue, maxValue)
    saveMeanMinMax(mean, minVal, maxVal, savePath, 'mean-min-max_' + name)

    saveProcessedDatasetHdf5(data, y, name='crossvalid_' + name, pathToWrite=savePath)


    f = open(savePath + 'paths_' + name + '.txt', 'w')
    [f.write(p + '\n') for p in paths]
    f.close()

    return [mean, minVal, maxVal]

def crossValidation():
    
    names = ['Acrima', 'Dristhi', 'Hrf',  'iChallenge-Gon', 'Jsiec', 'Rim-one-1', 'Rim-one-2', 'Rim-one-3']

    from sklearn.model_selection import StratifiedKFold, KFold

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def turn_img_bin(img, tres=0.5, maxValue=1.):
        t, img = cv2.threshold(img, tres, maxValue, cv2.THRESH_BINARY)
        return np.expand_dims(img, axis=2)

    x = load_hdf5('./train_dataset.hdf5').astype('float32')
    vessel = load_hdf5('./vessel/vessel_train_dataset.hdf5').astype('float32')
    vessel = [turn_img_bin(img) for img in vessel]
    vessel =np.array(vessel, 'float32')

    y = load_hdf5('./train_labels.hdf5')
    train_paths = getAllPaths('./Train_dataset/', names)

    i = 0
    for train, valid in kfold.split(train_paths, np.argmax(y, 1) ):
        print('=================== TRAIN ', i, ' ================')

        x_train = x[train]
        x_train = np.array(x_train, 'float32')

        r = saveCrossValidData(x_train, y[train],  [train_paths[k] for k in train], 'train_' + str(i))
        write_hdf5(vessel[train], './CrossValidData/vessel_train_' + str(i) + '.hdf5')

        #======================================================================
        print('=================== VALID', i, ' ================')

        mean, minVal, maxVal = r[0], r[1], r[2]

        saveCrossValidData(x[valid], y[valid], [train_paths[k] for k in valid], 'valid_' + str(i), mean, minVal, maxVal)
        write_hdf5(vessel[valid], './CrossValidData/vessel_valid_' + str(i) + '.hdf5')
        #'''

        i = i + 1

def loadMeanMinMax(path='./', name = "mean-min-max"):
    f = open(path + name + ".txt", "r")
    lines = f.readlines()
    f.close()

    meanTotal = np.array( [ float(v) for v in lines[:3]], dtype=np.float32 )
    minValue = float(lines[3])
    maxValue = float(lines[4])

    return meanTotal, minValue, maxValue
    
if __name__ == '__main__':
    x = load_hdf5('./test_dataset.hdf5')
    y = load_hdf5('./CrossValidData/crossValid_train_0_labels.hdf5')
    v = load_hdf5('./vessel/vessel_test_dataset.hdf5')

    ind = 200
    cv2.imshow('', v[ind])
    cv2.imshow('a', x[ind]/255.)
    cv2.waitKey(0)

    #'''

    '''
    x = load_hdf5('./CrossValidData/crossvalid_train_0_dataset.hdf5')
    img = cv2.imread('Train_dataset/Acrima/Glaucoma/Im310_g_ACRIMA.jpg')

    f = open('./CrossValidData/paths_train_0.txt', 'r')
    lines = f.readlines()
    f.close()

    x3 = []
    for l in lines:
        im = cv2.imread('./Train_dataset/' + l[:-1])
        im = cv2.resize(im, (112,112))
        x3.append(im)
    x3 = np.array(x3, dtype='float32')

    print(x3.shape, x.shape)
    print(type(x), type(x3))

    mean = calculateMeanDataset(x3)
    print(mean)

    mean = np.array([31.515855947600116, 83.57213246941646, 176.337470412811], 'float32')
    minv = -176.33746337890625
    maxv = 222.48414611816406

    data, m, mm, mmm = preprocess(x3.copy(), mean, minv, maxv)


    x4 = np.array([img], 'float32')
    x4, m, mm, mmm = preprocess(x4, mean, minv, maxv)
    print(x4[0])

    #'''




    '''
    cv2.imshow('a', x[ind])
    cv2.imshow('b', vessel[ind])
    cv2.moveWindow('a', 100, 100)
    cv2.moveWindow('b', 500, 100)
    cv2.waitKey(0)#'''
#'''