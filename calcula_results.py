from keras.utils import Sequence
from keras.utils import to_categorical
import numpy as np
from dataGenerator import *

from dataset_utils import *

from augmentation import *


import cv2
from sklearn.model_selection import train_test_split
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle

import os
os.environ['PYTHONHASHSEED'] = '0'

import random
from random import seed, randint
seed(1234)

import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.set_random_seed(4321)



from keras import backend as K

config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True

K.set_session(tf.Session(graph=tf.get_default_graph(), config=config))


from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session

# Reset Keras Session
def resetKeras():
    sess = get_session()
    clear_session()
    sess.close()

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(graph=tf.get_default_graph(), config=config))


from keras.utils import to_categorical
import gc
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
from keras.models import model_from_json
import h5py


from joblib import Parallel, delayed
import multiprocessing as mp
import time

#K.set_image_data_format('channels_last')

# ======================== MY CUSTOM METRICS =====================================================================================
def customMetrics(y_true, y_pred):
    from keras import backend as K
    y_true = K.clip(y_true, 0, 1)
    y_true = tf.one_hot(tf.argmax(y_true, axis = 1), 2)
    y_true_n = 1 - y_true

    y_pred = K.clip(y_pred, 0, 1)
    y_pred = tf.one_hot(tf.argmax(y_pred, axis = 1), 2)
    y_pred_n = 1 - y_pred

    result = K.sum(y_pred * y_true, 0)
    tp = result[1]
    tn = result[0]

    result = K.sum(y_pred_n * y_true, 0)
    fp = result[0]
    fn = result[1]

    dic = {
        'sens' : tp/(fn + tp),
        'spec' : tn/(tn + fp),
        'acc' : (tp + tn)/(tp + tn + fp + fn),
        'prec' : tp/(tp + fp + K.epsilon())
    }
    
    dic['f1'] = 2.*( tp/(tp + fp + K.epsilon()) * tp/(fn + tp))/(tp/(tp + fp + K.epsilon()) + tp/(fn + tp) + K.epsilon())

    return dic

def Acc(y_true, y_pred):
    return customMetrics(y_true, y_pred)['acc']

def Sens(y_true, y_pred):
    return customMetrics(y_true, y_pred)['sens']

def Spec(y_true, y_pred):
    return customMetrics(y_true, y_pred)['spec']

def Prec(y_true, y_pred):
    return customMetrics(y_true, y_pred)['prec']

def F1(y_true, y_pred):
    return customMetrics(y_true, y_pred)['f1']
# ========================================================================================================================================
def customMetricsNP(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    try:
        tn, fp, fn, tp = confusion_matrix(np.argmax(y_true, 1), np.argmax(y_pred, 1)).ravel()
    except:
        from keras.utils import to_categorical
        y_true = np.clip(y_true, 0, 1)
        y_true = to_categorical(np.argmax(y_true, axis = 1), 2)
        y_true_n = 1 - y_true

        y_pred = np.clip(y_pred, 0, 1)
        y_pred = to_categorical(np.argmax(y_pred, axis = 1), 2)
        y_pred_n = 1 - y_pred

        result = np.sum(y_pred * y_true, 0)
        tp = result[1]
        tn = result[0]

        result = np.sum(y_pred_n * y_true, 0)
        fp = result[0]
        fn = result[1]#'''

    dic = {
        'tp' : tp,
        'tn' : tn,
        'fp' : fp,
        'fn' : fn,
        'sens' : tp/(fn + tp),
        'spec' : tn/(tn + fp),
        'acc' : (tp + tn)/(tp + tn + fp + fn),
        'prec' : tp/(tp + fp) if tp + fp > 0 else 0.
    }
    
    dic['f1'] = 2.*(dic['prec'] * dic['sens'])/(dic['prec'] + dic['sens']) if (dic['prec'] + dic['sens']) > 0 else 0.

    return dic
# =========================================================================================================================================

'''
def preprocess(img):
    from keras.applications.vgg16 import preprocess_input

    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess_input(img.astype(float), mode = 'tf')#'''



def processData(data):
    newdata = np.empty( (len(data), 14, 14, 512) )

    from keras.applications.vgg16 import VGG16
    from keras.models import Sequential

    # loading vgg16
    vgg16_model = VGG16(include_top = False, input_shape = (224,224,3))

    for layer in vgg16_model.layers[:-4]:
        layer.trainable = False
    my_model = Sequential()
    for l in vgg16_model.layers[:-4]:
        my_model.add(l) 

    for i, d in enumerate(data):
        #img = preprocess(d)
        img = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
        img = my_model.predict(img.reshape(1, 224, 224, 3))#'''
        newdata[i,] = img
    
    return newdata



def preprocess(x, meanTotal=None, minValue=None, maxValue=None):
    if meanTotal is not None: meanTotal = calculateMeanDataset(x)
    x = extractMean(x, meanTotal)

    if minValue is not None: minValue = np.min(x)
    if maxValue is not None: maxValue = np.max(x)

    x = normalizeDataset(x, minValue, maxValue)

    print('\n==============', meanTotal, minValue, maxValue, '\n')
    return x, meanTotal, minValue, maxValue


from keras.initializers import VarianceScaling, RandomNormal, RandomUniform, TruncatedNormal, glorot_uniform 

def get_model(space):
    from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input
    from keras.models import Model

    inputs = Input(shape=(14,14,512), name = 'input_layer')

    x = Conv2D(kernel_size = (space['kernel_size'],space['kernel_size']), strides = (1,1), filters = int(512 * space['filters_mult']), bias_initializer = 'zeros', activation = 'relu', kernel_initializer = TruncatedNormal(seed = 42), 
             padding = 'same', name = 'block1_conv1',trainable = True, use_bias = True )(inputs)

    for i in range(1, 3):
        x =  Conv2D(kernel_size = (space['kernel_size'],space['kernel_size']), strides = (1,1), filters = int(512 * space['filters_mult']), bias_initializer = 'zeros', activation = 'relu', kernel_initializer = TruncatedNormal(seed = 42 * i), 
                padding = 'same', name = 'block1_conv' + str(i+1),trainable = True, use_bias = True )(x)

    x = MaxPool2D(name = 'block1_pool')(x)

    x = Flatten(name = 'flatten')(x)

    for i in range(0,space['num_units']):
        x = Dense(int(4096 * space['units_mult']), activation = space['activation'], name = 'fc' + str(i + 1),   #kernel_constraint=maxnorm(space['kernel_constraint'])
             kernel_initializer = glorot_uniform(seed=84), trainable = True, use_bias = True )(x)
        x = Dropout(space['dropout'], seed = 42)(x)

    pred = Dense(2, activation = 'softmax', name = 'predictions', kernel_initializer = 'zeros')(x)
    model = Model(inputs, pred)

    return model

def getOptimizer(opt, learnRate):
    from keras.optimizers import Adamax, Adadelta, Adam, SGD, Nadam, Adagrad
    if opt == 'adam':
        return Adam(lr = learnRate)
    else: 
        if opt == 'sgd':
            return SGD(lr = learnRate)
        else:
            if opt == 'adadelta':
                return Adadelta(lr = learnRate)
            else:
                if opt == 'adamax':
                    return Adamax(lr = learnRate)
                else:
                    if opt == 'nadam':
                        return Nadam(lr = learnRate)
                    else:
                        if opt == 'adagrad':
                            return Adagrad(lr = learnRate)


def saveModel(model, pathToSave, modelName):
    # serialize model to JSON
    model_json = model.to_json()
    with open(pathToSave + '/' + modelName + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(pathToSave + '/' + modelName + ".h5")


def load_trained_model(jsonfile, weightsfile):
    from keras.models import model_from_json
    # load json and create model
    json_file = open(jsonfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weightsfile)
    print("Loaded model from disk")
    return loaded_model#'''


def train_model(args):
    from keras.models import Sequential
    from keras.utils import Sequence
    from keras.applications.vgg16 import VGG16
    from keras.models import Sequential, Model
    from keras.callbacks import EarlyStopping, LearningRateScheduler, CSVLogger
    from keras.optimizers import Adamax, Adadelta, Adam
    from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D

    #x = args["x"]
    #y = args["y"]
    x_train = args["x_train"]
    y_train = args["y_train"]
    x_valid = args["x_valid"]
    y_valid = args["y_valid"]
    x_test = args["x_test"]
    y_test = args["y_test"]
    space = args["space"]
    save_dir = args["save_dir"]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = get_model(space)

    early = EarlyStopping(monitor='val_Sens', min_delta=0.03, patience=int(0.7 * space['epochs']), verbose=0, mode='max')
    lr_decay = LearningRateScheduler(schedule=lambda epoch: space['lr'] * (space['lr_decay'] ** epoch))
    log = CSVLogger(save_dir + '/log.csv')

    model.compile(loss=space['loss'],
              optimizer=getOptimizer(space['optimizer'], space['lr']),
              metrics=['accuracy', Sens, Spec, Prec, F1])

    paramTrainGenerator = {
        'dim' : (14,14), 
        'batch_size' : space['batch_size'], 
        'n_classes' : 2,  
        'n_channels' : 512,
        'shuffle' : True, 
        'data' : x_train
    }

    paramValidGenerator = {
        'dim' : (14,14), 
        'batch_size' : space['batch_size'], 
        'n_classes' : 2,  
        'n_channels' : 512,
        'shuffle' : True, 
        'data' : x_valid
    }

    # gerador de batches das imagens de treino
    training_generator = DataGenerator(np.asarray(range(0, len(x_train))), y_train, **paramTrainGenerator)

    # gerador de batches das imagens de validacao
    valid_generator = DataGenerator(np.asarray(range(0, len(x_valid))), y_valid, **paramValidGenerator)

    try:
        # Fit the model
        history = model.fit_generator(generator=training_generator, steps_per_epoch = len(x_train) / paramTrainGenerator['batch_size'], 
                validation_data=valid_generator, validation_steps = len(x_valid) / paramValidGenerator['batch_size'], 
                epochs = space['epochs'], verbose = 2, callbacks = [early, log])
    except:
        return -1


    y_pred = model.predict(x_test)
    metricsTest = customMetricsNP(to_categorical(y_test, 2), y_pred)

    y_pred = model.predict(x_valid)
    metricsValid = customMetricsNP(to_categorical( np.argmax(y_valid, 1), 2), y_pred)

    y_pred = model.predict(x_train)
    metricsTrain = customMetricsNP(to_categorical( np.argmax(y_train, 1), 2), y_pred)


    # saving the history
    with open(args['save_dir']+'history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # saving model
    if metricsValid['sens'] != 0. and metricsValid['spec'] != 0. and metricsTest['sens'] != 0. and metricsTest['spec'] != 0.: 
        saveModel(model, save_dir, 'final_model')



    return {
    	"train" : metricsTrain,
    	"valid" : metricsValid,
    	"test" : metricsTest
    }





# =============== CROSS VALIDATION ================================================================================================


def saveCrossValidData(x, y, paths, name='', meanTotal=None, minValue=None, maxValue=None):
    savePath = './CrossValidData/'
    if not os.path.exists(savePath): os.makedirs(savePath)

    data, mean, minVal, maxVal = preprocess(x, meanTotal, minValue, maxValue)
    saveMeanMinMax(mean, minVal, maxVal, savePath, 'mean-min-max_' + name)

    data = processData(data)

    saveProcessedDatasetHdf5(data, y, name='crossvalid_' + name, pathToWrite=savePath)

    f = open(savePath + 'paths_' + name + '.txt', 'w')
    [f.write(p + '\n') for p in paths]
    f.close()

    return [mean, minVal, maxVal]


def aux(args):
    return saveCrossValidData(args['x'], args['y'], args['paths'], args['name'])

def aux2(args):
    return saveCrossValidData(args['x'], args['y'], args['paths'], args['name'], args['mean'], args['min'], args['max'])

def crossValidation():
    
    names = ['Acrima', 'Hrf', 'Jsiec', 'Drishti', 'iChallenge-Gon', 'Rim-one-1', 'Rim-one-2', 'Rim-one-3']

    from sklearn.model_selection import StratifiedKFold, KFold

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    x = load_hdf5('./train_dataset.hdf5')
    y = load_hdf5('./train_labels.hdf5')
    train_paths = getAllPaths('./Train_dataset/', names)

    i = 0
    for train, valid in kfold.split(train_paths, np.argmax(y, 1) ):
        print('=================== TRAIN ', i, ' ================')
        args = {
            'x' : x[train],
            'y' : y[train],
            'paths' : [train_paths[k] for k in train],
            'name' : 'train_' + str(i)
        }

        pool = mp.Pool(1)
        r = pool.map(aux, [args] )

        pool.close()
        pool.join()
        #======================================================================
        print('=================== VALID', i, ' ================')

        mean, minVal, maxVal = r[0][0], r[0][1], r[0][2]

        args = {
            'x' : x[valid],
            'y' : y[valid],
            'paths' : [train_paths[k] for k in valid],
            'name' : 'valid_' + str(i),
            'mean' : mean,
            'min' : minVal,
            'max' : maxVal
        }

        pool = mp.Pool(1)
        r = pool.map(aux2, [args] )

        pool.close()
        pool.join()#'''

        i = i + 1
####################################################






#=================================================================================================================================
global x, y, data, labels, paths

global x_test, y_test, data_test, labels_test, paths_test
#=================================================================================================================================


def objective(space):
    from sklearn.model_selection import StratifiedKFold, KFold

    global x, y, data, labels, paths
    global x_test, y_test, data_test, labels_test, paths_test

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # will store the results (acc, sens, spec, f1) from all 10 models trained
    allResults = []

    # these will store all metrics individually (to calculate the mean and standard deviation)
    acc = []
    sens = []
    spec = []
    f1 = []

    args = {
    "x" : x,
    "y" : y,
    "x_test" : x_test,
    "y_test" : y_test,
    "space" : space
    }

    # training the model using cross validation
    for train, valid in kfold.split(paths, y):
        args["x_train"] = x[train]
        args["y_train"] = y[train]
        args["x_valid"] = x[valid]
        args["y_valid"] = y[valid]

        print('=================== SPACE ===============================')
        print(space)

        # training the model in a separated process and getting the results
        pool = mp.Pool(1)
        r = pool.map(train_model, [args] )

        pool.close()
        pool.join()

        allResults.append(r[0])
        
        if r[0] == -1:
            pool.close()
            return {
                'loss' : -0,
                'status' : STATUS_OK,
                'metrics' : None,
                'results' : allResults,
                'space' : space,
                'eval_time' : time.time()
            }

        r = r[0]

        print("==================== RESULT ============================")
        print("===== TEST ========")
        print(r['test'])
        print("===== TRAIN =======")
        print(r['train'])
        print("===== VALID =======")
        print(r['valid'])

        acc.append(r["test"]["acc"])
        sens.append(r["test"]["sens"])
        spec.append(r["test"]["spec"])
        f1.append(r["test"]["f1"])

        gc.collect()

        if r['test']['sens'] == 0 or r['test']['spec'] == 0: break


    metrics = {
        "acc_mean" : np.mean(acc),
        "spec_mean" : np.mean(spec),
        "sens_mean" : np.mean(sens),
        "f1_mean" : np.mean(f1),

        "acc_std" : np.std(acc),
        "spec_std" : np.std(spec),
        "sens_std" : np.std(sens),
        "f1_std" : np.std(f1)
    }


    print("================ FINAL RESULT ==========================")
    print('Acuracia: ' + str(metrics["acc_mean"]) +  ' -- ' + str(metrics['acc_std']) )
    print('Sens:'+ str(metrics["sens_mean"]) +  ' -- ' + str(metrics['sens_std']) )
    print('Spec:' + str(metrics["spec_mean"]) +  ' -- ' + str(metrics['spec_std']) )
    print('F1:' + str(metrics["f1_mean"]) +  ' -- ' + str(metrics['f1_std']) )
    

    return {
        'loss' : -metrics["acc_mean"],
        'status' : STATUS_OK,
        'metrics' : metrics,
        'results' : allResults,
        'space' : space,
        'eval_time' : time.time()
    }




space = {
    'batch_size' : hp.choice('batch_size', np.arange(25, 50, 5, dtype = int) ),
    'lr' : hp.loguniform('lr', -11.51292, -6.90775),
    'lr_decay' : hp.loguniform('lr_decay', -2.30258, -0.10536),
    'optimizer' : hp.choice('optimizer', ['adadelta', 'adam', 'adamax', 'nadam']),
    'activation' : hp.choice('activation', ['relu', 'tanh', 'elu']),
    'dropout' : hp.uniform('dropout', 0.1, 0.5),
    'units_mult' : hp.uniform('units_mult', 0.3, 1.0),
    'epochs' : hp.choice('epochs',  np.arange(40, 85, 5, dtype = int)),
    'num_units' : hp.choice('num_units', [2,3,4]),
    'kernel_size' : hp.choice('kernel_size', [5,7,9,11]),
    'filters_mult' : hp.uniform('filters_mult', 0.5, 1.2),
    'loss' : hp.choice('loss', ['categorical_crossentropy', 'squared_hinge', 'categorical_hinge'])
}

space = {
    'batch_size' : 25,
    'lr' : 2.271053530156096e-05,
    'lr_decay' : 0.10017310799319072,
    'optimizer' : 'adam',
    'activation' : 'relu',
    'dropout' : 0.15858232959401378,
    'units_mult' : 0.5496216821295921,
    'epochs' : 45,
    'num_units' : 4,
    'kernel_size' : 5,
    'filters_mult' : 0.984985087426544,
    'loss' : 'squared_hinge'
}#'''


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


# Normalize a dataset using min-max normalization (let the values between 0 an 1)
def normalizeDataset(x, minValue=None, maxValue=None):
    if minValue == None: minValue = np.min(x)
    if maxValue == None: maxValue = np.max(x)

    print('MIN_MAX: ', minValue, maxValue)

    return (x - minValue)/(maxValue - minValue)


def saveMeanMinMax(meanTotal, minValue, maxValue, save_dir = './', name = 'mean-min-max'):
    file = open(save_dir + name + ".txt", "w")
    [ file.write(str(v) + '\n') for v in meanTotal ]
    file.write(str(minValue) + '\n')
    file.write(str(maxValue))
    file.close()


def loadMeanMinMax(path='./', name = "mean-min-max"):
    f = open(path + name + ".txt", "r")
    lines = f.readlines()
    f.close()

    meanTotal = np.array( [ float(v) for v in lines[:3]], dtype=np.float32 )
    minValue = float(lines[3])
    maxValue = float(lines[4])

    return meanTotal, minValue, maxValue


model_number = 0
def objective2(space):
    global model_number
    model_number = model_number + 1

    print('==================================  INIT MODEL NUMBER ', model_number, ' =============================================')
    # will store the results (acc, sens, spec, f1) from all 10 models trained
    allResults = []

    # these will store all metrics individually (to calculate the mean and standard deviation)
    acc_test = []
    sens_test = []
    spec_test = []
    f1_test = []

    acc_valid = []
    sens_valid = []
    spec_valid = []
    f1_valid = []

    acc_train = []
    sens_train = []
    spec_train = []
    f1_train = []

    args = {
        "space" : space
    }

    

    for i in range(10):
        s = str(i)
        print('============================== MODEL ', model_number, ' - TESTE ', s, ' =======================================')
        args["x_train"] = load_hdf5('./CrossValidData/crossvalid_train_' + s + '_dataset.hdf5')
        args["y_train"] = load_hdf5('./CrossValidData/crossvalid_train_' + s + '_labels.hdf5')
        args["x_valid"] = load_hdf5('./CrossValidData/crossvalid_valid_' + s + '_dataset.hdf5')
        args["y_valid"] = load_hdf5('./CrossValidData/crossvalid_valid_' + s + '_labels.hdf5')
        args["x_test"] = load_hdf5('./test_dataset.hdf5')
        args["y_test"] = load_hdf5('./test_labels.hdf5')
        args["y_test"] = np.argmax(args["y_test"], 1)

        mean, minVal, maxVal = loadMeanMinMax('./CrossValidData/', 'mean-min-max_train_' + s)
        args["x_test"], mean, minVal, maxVal = preprocess(args["x_test"], mean, minVal, maxVal)

        #args["x_train"] , args["y_train"] = balancedDataAugmentation(args["x_train"], args["y_train"])

        pool = mp.Pool(1)
        r = pool.map(processData, [args["x_test"]] )
        pool.close()
        pool.join()
        args['x_test'] = r[0]

        args["space"] = space
        args["save_dir"] = './result/modelo_' + str(model_number) + '/teste_' + s + '/'
        if not os.path.exists('./result/modelo_' + str(model_number) + '/'): os.makedirs('./result/modelo_' + str(model_number) + '/')
        
        print('=================== SPACE ===============================')
        print(space)



        # training the model in a separated process and getting the results
        pool = mp.Pool(1)
        r = pool.map(train_model, [args] )

        pool.close()
        pool.join()

        allResults.append(r[0])
        
        if r[0] == -1:
            pool.close()
            return {
                'loss' : -0.,
                'status' : STATUS_OK,
                'result_test' : None,
                'result_train' : None,
                'result_valid' : None,
                'result' : None,
                'space' : space,
                'eval_time' : time.time()
            }

        r = r[0]

        print("==================== RESULT ============================")
        print("===== TEST ========")
        print(r['test'])
        print("===== TRAIN =======")
        print(r['train'])
        print("===== VALID =======")
        print(r['valid'])

        acc_test.append(r["test"]["acc"])
        sens_test.append(r["test"]["sens"])
        spec_test.append(r["test"]["spec"])
        f1_test.append(r["test"]["f1"])

        acc_valid.append(r["valid"]["acc"])
        sens_valid.append(r["valid"]["sens"])
        spec_valid.append(r["valid"]["spec"])
        f1_valid.append(r["valid"]["f1"])

        acc_train.append(r["train"]["acc"])
        sens_train.append(r["train"]["sens"])
        spec_train.append(r["train"]["spec"])
        f1_train.append(r["train"]["f1"])

        gc.collect()

        if i == 0 and (r['test']['sens'] == 0 or r['test']['spec'] == 0): break


    train_metrics = {
        "acc_mean" : np.mean(acc_train),
        "spec_mean" : np.mean(spec_train),
        "sens_mean" : np.mean(sens_train),
        "f1_mean" : np.mean(f1_train),

        "acc_std" : np.std(acc_train),
        "spec_std" : np.std(spec_train),
        "sens_std" : np.std(sens_train),
        "f1_std" : np.std(f1_train),

        "acc_max" : np.max(acc_train),
        "spec_max" : np.max(spec_train),
        "sens_max" : np.max(sens_train),
        "f1_max" : np.max(f1_train),

        "acc_min" : np.min(acc_train),
        "spec_min" : np.min(spec_train),
        "sens_min" : np.min(sens_train),
        "f1_min" : np.min(f1_train)
    }

    valid_metrics = {
        "acc_mean" : np.mean(acc_valid),
        "spec_mean" : np.mean(spec_valid),
        "sens_mean" : np.mean(sens_valid),
        "f1_mean" : np.mean(f1_valid),

        "acc_std" : np.std(acc_valid),
        "spec_std" : np.std(spec_valid),
        "sens_std" : np.std(sens_valid),
        "f1_std" : np.std(f1_valid),

        "acc_max" : np.max(acc_valid),
        "spec_max" : np.max(spec_valid),
        "sens_max" : np.max(sens_valid),
        "f1_max" : np.max(f1_valid),

        "acc_min" : np.min(acc_valid),
        "spec_min" : np.min(spec_valid),
        "sens_min" : np.min(sens_valid),
        "f1_min" : np.min(f1_valid)
    }

    test_metrics = {
        "acc_mean" : np.mean(acc_test),
        "spec_mean" : np.mean(spec_test),
        "sens_mean" : np.mean(sens_test),
        "f1_mean" : np.mean(f1_test),

        "acc_std" : np.std(acc_test),
        "spec_std" : np.std(spec_test),
        "sens_std" : np.std(sens_test),
        "f1_std" : np.std(f1_test),

        "acc_max" : np.max(acc_test),
        "spec_max" : np.max(spec_test),
        "sens_max" : np.max(sens_test),
        "f1_max" : np.max(f1_test),

        "acc_min" : np.min(acc_test),
        "spec_min" : np.min(spec_test),
        "sens_min" : np.min(sens_test),
        "f1_min" : np.min(f1_test)
    }

    print("================ FINAL RESULT ==========================")
    print('====================  TESTE ===============================')
    print('Acuracia: ' + str(test_metrics["acc_mean"]) +  ' -- ' + str(test_metrics['acc_std']) )
    print('Sens:'+ str(test_metrics["sens_mean"]) +  ' -- ' + str(test_metrics['sens_std']) )
    print('Spec:' + str(test_metrics["spec_mean"]) +  ' -- ' + str(test_metrics['spec_std']) )
    print('F1:' + str(test_metrics["f1_mean"]) +  ' -- ' + str(test_metrics['f1_std']) )

    print('====================  VALIDACAO ===============================')
    print('Acuracia: ' + str(valid_metrics["acc_mean"]) +  ' -- ' + str(valid_metrics['acc_std']) )
    print('Sens:'+ str(valid_metrics["sens_mean"]) +  ' -- ' + str(valid_metrics['sens_std']) )
    print('Spec:' + str(valid_metrics["spec_mean"]) +  ' -- ' + str(valid_metrics['spec_std']) )
    print('F1:' + str(valid_metrics["f1_mean"]) +  ' -- ' + str(valid_metrics['f1_std']) )
    

    return {
        'loss' : -valid_metrics["acc_mean"],
        'status' : STATUS_OK,
        'result_test' : test_metrics,
        'result_train' : train_metrics,
        'result_valid' : valid_metrics,
        'result' : allResults,
        'space' : space,
        'eval_time' : time.time()
    }


def saveWrongOnes(testNumber = 0):
    names = ['Acrima', 'Hrf', 'Jsiec', 'Drishti', 'iChallenge-Gon', 'Rim-one-1', 'Rim-one-2', 'Rim-one-3']

    s = str(testNumber)

    x = load_hdf5('./test_dataset.hdf5')
    y = load_hdf5('./test_labels.hdf5')
    paths = getAllPaths('./Test_dataset/', names)


    mean, minVal, maxVal = loadMeanMinMax('./CrossValidData/', 'mean-min-max_train_' + s)
    x_processed, mean, minVal, maxVal = preprocess(x.copy(), mean, minVal, maxVal)


    pool = mp.Pool(1)
    r = pool.map(processData, [x_processed] )
    pool.close()
    pool.join()
    x_processed = r[0]


    model = load_trained_model('./result/modelo_74/teste_' + s + '/final_model.json', './result/modelo_74/teste_' + s + '/final_model.h5')


    y_pred = model.predict(x_processed)
    metrics= customMetricsNP(y, y_pred)

    print('=== METRICAS: ', metrics)

    
    y2 = np.argmax(y, 1)
    wrong = []
    for c, lab in enumerate(np.argmax(y_pred, 1)) :
        if lab != y2[c]:
            wrong.append(c)
            
    
    if not os.path.exists('./wrongs/'): os.makedirs('./wrongs/')

    for n in names:
        if not os.path.exists('./wrongs_teste_' + s + '/' + n): os.makedirs('./wrongs_teste_' + s + '/' + n)
        if not os.path.exists('./wrongs_teste_' + s + '/' + n + '/Glaucoma/'): os.makedirs('./wrongs_teste_' + s + '/' + n + '/Glaucoma/')
        if not os.path.exists('./wrongs_teste_' + s + '/' + n + '/Non-Glaucoma/'): os.makedirs('./wrongs_teste_' + s + '/' + n + '/Non-Glaucoma/')

    for w in wrong:
        cv2.imwrite('./wrongs_teste_' + s + '/' + paths[w], x[w])#'''

    file = open('./wrongs_teste_' + s + '/metrics.txt', 'w')
    file.write('sens: ' +  str(metrics['sens']) + '\n')
    file.write('spec: ' +  str(metrics['spec']) + '\n')
    file.write('prec: ' +  str(metrics['prec']) + '\n')
    file.write('f1: ' +  str(metrics['f1']) + '\n')
    file.write('acc: ' +  str(metrics['acc']) + '\n')
    file.write('tp: ' +  str(metrics['tp']) + '\n')
    file.write('tn: ' +  str(metrics['tn']) + '\n')
    file.write('fp: ' +  str(metrics['fp']) + '\n')
    file.write('fn: ' +  str(metrics['fn']) + '\n')
    file.close()

    
    ind =[len(os.listdir('./Test_dataset/' + n + '/Glaucoma/')) + len(os.listdir('./Test_dataset/' + n + '/Non-Glaucoma/')) for n in names]
    ind = np.array(ind)

    w_ind = [len(os.listdir('./wrongs_teste_' + s + '/' + n + '/Glaucoma/')) + len(os.listdir('./wrongs_teste_' + s + '/' + n + '/Non-Glaucoma/')) for n in names]

    [print(c, '-', n, ':', ind[c], '==', w_ind[c]) for c, n in enumerate(names)]


    file = open('./wrongs_teste_' + s + '/metrics_individual.txt', 'w')

    for base in range(len(ind)):
        file.write('=============== ' + names[base] +' ===============\n')
        soma = 0
        
        if base != 0:
            soma = np.sum( ind[: int(base) ] )
        init = soma
        final = soma + ind[int(base)]

        print(init, final)

        y_pred = model.predict(x_processed[init:final])
        metrics= customMetricsNP(y[init:final], y_pred)

        print('=== METRICAS: ', metrics)#'''

        file.write('  sens: ' +  str(metrics['sens']) + '\n')
        file.write('  spec: ' +  str(metrics['spec']) + '\n')
        file.write('  prec: ' +  str(metrics['prec']) + '\n')
        file.write('  f1: ' +  str(metrics['f1']) + '\n')
        file.write('  acc: ' +  str(metrics['acc']) + '\n')
        file.write('  tp: ' +  str(metrics['tp']) + '\n')
        file.write('  tn: ' +  str(metrics['tn']) + '\n')
        file.write('  fp: ' +  str(metrics['fp']) + '\n')
        file.write('  fn: ' +  str(metrics['fn']) + '\n')
    file.close()


def saveWrongOnesValid(testNumber = 0):

    names = ['Acrima', 'Hrf', 'Jsiec', 'Drishti', 'iChallenge-Gon', 'Rim-one-1', 'Rim-one-2', 'Rim-one-3']

    s = str(testNumber)

    x_processed = load_hdf5('./CrossValidData/crossvalid_valid_' + s + '_dataset.hdf5')
    y = load_hdf5('./CrossValidData/crossvalid_valid_' + s + '_labels.hdf5')

    file = open('./CrossValidData/paths_valid_' + s + '.txt', 'r')
    paths = file.readlines()
    file.close()

    paths = [str(p).replace('\n', '') for p in paths]


    model = load_trained_model('./result/modelo_74/teste_' + s + '/final_model.json', './result/modelo_74/teste_' + s + '/final_model.h5')


    y_pred = model.predict(x_processed)
    metrics= customMetricsNP(y, y_pred)

    print('=== METRICAS: ', metrics)

    
    y2 = np.argmax(y, 1)
    wrong = []
    for c, lab in enumerate(np.argmax(y_pred, 1)) :
        if lab != y2[c]:
            wrong.append(c)
            
    
    for n in names:
        if not os.path.exists('./wrongs_valid_' + s + '/' + n): os.makedirs('./wrongs_valid_' + s + '/' + n)
        if not os.path.exists('./wrongs_valid_' + s + '/' + n + '/Glaucoma/'): os.makedirs('./wrongs_valid_' + s + '/' + n + '/Glaucoma/')
        if not os.path.exists('./wrongs_valid_' + s + '/' + n + '/Non-Glaucoma/'): os.makedirs('./wrongs_valid_' + s + '/' + n + '/Non-Glaucoma/')

    print('=====SHPAE ===', x_processed.shape)
    for w in wrong:
        img = cv2.imread('./Train_dataset/' + paths[w])
        b = cv2.imwrite('./wrongs_valid_' + s + '/' + paths[w], img)#'''

    file = open('./wrongs_valid_' + s + '/metrics.txt', 'w')
    file.write('sens: ' +  str(metrics['sens']) + '\n')
    file.write('spec: ' +  str(metrics['spec']) + '\n')
    file.write('prec: ' +  str(metrics['prec']) + '\n')
    file.write('f1: ' +  str(metrics['f1']) + '\n')
    file.write('acc: ' +  str(metrics['acc']) + '\n')
    file.write('tp: ' +  str(metrics['tp']) + '\n')
    file.write('tn: ' +  str(metrics['tn']) + '\n')
    file.write('fp: ' +  str(metrics['fp']) + '\n')
    file.write('fn: ' +  str(metrics['fn']) + '\n')
    file.close()

    
    ind = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    for i in range(8):
        ind[i] = np.sum([ 1 for p in paths if p.split('/')[0] == names[i] ])

    [ print(n, ':', ind[c]) for c, n in enumerate(names) ]


    file = open('./wrongs_valid_' + s + '/metrics_individual.txt', 'w')
    
    for base in range(len(ind)):
        if ind[base] == 0:
            continue
        file.write('=============== ' + names[base] +' ===============\n')
        soma = 0
        
        if base != 0:
            soma = np.sum( ind[: int(base) ] )
        init = soma
        final = soma + ind[int(base)]

        print(init, final)

        y_pred = model.predict(x_processed[init:final])
        metrics= customMetricsNP(y[init:final], y_pred)

        print('=== METRICAS: ', metrics)

        file.write('  sens: ' +  str(metrics['sens']) + '\n')
        file.write('  spec: ' +  str(metrics['spec']) + '\n')
        file.write('  prec: ' +  str(metrics['prec']) + '\n')
        file.write('  f1: ' +  str(metrics['f1']) + '\n')
        file.write('  acc: ' +  str(metrics['acc']) + '\n')
        file.write('  tp: ' +  str(metrics['tp']) + '\n')
        file.write('  tn: ' +  str(metrics['tn']) + '\n')
        file.write('  fp: ' +  str(metrics['fp']) + '\n')
        file.write('  fn: ' +  str(metrics['fn']) + '\n')
    file.close()#'''


def testando():
    paths = []

    for i in range(10):
        file = open('./CrossValidData/paths_valid_' + str(i) + '.txt', 'r')
        paths = paths + file.readlines()
        file.close()

    paths = [str(p).replace('\n', '') for p in paths]

    c = 0
    for p in paths:
        if p.split('/')[0] == 'Jsiec':
            print(c, p)
            c = c + 1

if __name__ == "__main__":
    x = load_hdf5('./test_dataset.hdf5')
    y = load_hdf5('./test_labels.hdf5')

    mean = calculateMeanDataset(x)
    print(mean)






    



    '''
    args = {}

    args["x_train"] = load_hdf5('./CrossValidData/crossvalid_train_0_dataset.hdf5')
    args["y_train"] = load_hdf5('./CrossValidData/crossvalid_train_0_labels.hdf5')
    args["x_valid"] = load_hdf5('./CrossValidData/crossvalid_valid_0_dataset.hdf5')
    args["y_valid"] = load_hdf5('./CrossValidData/crossvalid_valid_0_labels.hdf5')

    args["x_test"] = load_hdf5('./test_dataset.hdf5')
    args["y_test"] = load_hdf5('./test_labels.hdf5')

    mean, minVal, maxVal = loadMeanMinMax('./CrossValidData/', 'mean-min-max_train_0')
    args["x_test"], mean, minVal, maxVal = preprocess(args["x_test"], mean, minVal, maxVal)

    pool = mp.Pool(1)
    r = pool.map(processData, [args["x_test"]] )
    pool.close()
    pool.join()

    args['x_test'] = r[0]
    args["y_test"] = np.argmax(args["y_test"], 1)

    args["space"] = space
    args["save_dir"] = './result'

    train_model(args)#'''

    



    '''
    x = load_hdf5('./dataset.hdf5')
    y = load_hdf5('./labels.hdf5')

    print(x.shape)

    meanTotal, minValue, maxValue =  loadMeanMinMax()

    x = extractMean(x, meanTotal)
    x = normalizeDataset(x, minValue, maxValue)
    #cv2.imshow('', x[0]/255.)
    #cv2.waitKey(0)




    
    im = cv2.resize(
        cv2.imread('D:/JoaoMarcello/Documents/CNPQ/Program/2020/project_dataset/Processed_dataset/Acrima/Glaucoma/Im310_g_ACRIMA.jpg'),
        (112,112)
    )

    print(x[0])
    print('------------------------------------------------------')

    x2 = extractMean(np.array([im], dtype=np.float32), meanTotal)
    x2 = normalizeDataset(x2, minValue=minValue, maxValue=maxValue)
    print(x2[0])#'''


    '''
    names = ['Acrima', 'Hrf', 'Jsiec', 'Drishti', 'iChallenge-Gon', 'Rim-one-1', 'Rim-one-2', 'Rim-one-3']
    x2, y2, paths = load_glaucoma_dataset('D:/JoaoMarcello/Documents/CNPQ/Program/2020/project_dataset/Processed_dataset/', names, (112,112))

    print('------------------------------------------------------')
    print(x2[0])#'''

    #saveProcessedDatasetHdf5(x2, y2, pathToWrite='./result')





    '''
    from sklearn.model_selection import StratifiedKFold, KFold

    global x, y, data, labels, paths
    global x_test, y_test, data_test, labels_test, paths_test
    
    # loading the development dataset (873 images / DRISHTI, RIM-ONE-1, RIM-ONE-2, RIM-ONE-3)
    x, y, data, labels, paths = loadDataset('D:/JoaoMarcello/Documents/Datasets/Train_Dataset/')

    # loading the test dataset (802 images / HRF, JSIEC, ACRIMA)
    x_test, y_test, data_test, labels_test, paths_test = loadDataset('D:/JoaoMarcello/Documents/Datasets/Test_Dataset/')

    print('============== PROCESSING X_TRAIN ===================')
    pool = mp.Pool(1)
    r = pool.map(processData, [x] )
    x = r[0]
    pool.close()
    pool.join()

    print('============== PROCESSING X_TEST ====================')
    pool = mp.Pool(1)
    r = pool.map(processData, [x_test] )
    x_test = r[0]
    pool.close()
    pool.join()

    del r
    gc.collect()#'''



    '''
    space = {
    'batch_size' : 10,
    'lr' :0.000040231,
    'lr_decay' : 0.9,
    'optimizer' :'adamax',
    'activation' : 'tanh',
    'dropout' :  0.45710103929561785,
    'units_mult' : 0.6293308305355723,
    'epochs' : 10,
    'num_units' : 3,
    'kernel_size' : 9,
    'filters_mult' : 0.675428418128738,
    'loss' : 'categorical_crossentropy'
    }

    objective(space)#'''


    '''
    if not os.path.exists('./result/'):
        os.makedirs('./result/')

    for i in range(1,2):
        try:
            trials = pickle.load(open("./result/" + "teste" + "_trials.p", "rb"))
            max_evals = len(trials.trials) + 4
        except FileNotFoundError:
            trials = Trials()
            max_evals = 1

        best = fmin(objective,
        space=space,
        algo=tpe.suggest,
        max_evals= max_evals,
        trials=trials)  #rstate= np.random.RandomState(400 * i)

        pickle.dump(trials, open("./result/" + "teste" + "_trials.p", "wb"))

        gc.collect()#'''




    '''
    # loading the development dataset (873 images / DRISHTI, RIM-ONE-1, RIM-ONE-2, RIM-ONE-3)
    x, y, data, labels, paths = loadDataset('D:/JoaoMarcello/Documents/Datasets/Train_Dataset/')

    # loading the test dataset (802 images / HRF, JSIEC, ACRIMA)
    x_test, y_test, data_test, labels_test, paths_test = loadDataset('D:/JoaoMarcello/Documents/Datasets/Test_Dataset/')

    args = {
    	"x" : x,
    	"y" : y,
    	"x_test" : x_test,
    	"y_test" : y_test,
    	"labels" : labels,
    	"space" : None
    }

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    allResults = []
    acc = []
    sens = []
    spec = []
    f1 = []
    
    for train, valid in kfold.split(paths, y):
        args["x_train"] = x[train]    
        args["y_train"] = y[train]
        args["x_valid"] = x[valid]
        args["y_valid"] = y[valid]
        
        pool = mp.Pool(1)
        r = pool.map(train_model, [args] )
        allResults.append(r[0])

        pool.close()
        pool.join()

        r = r[0]
        print("==================== RESULT ============================")
        print("===== TEST ========")
        print(r['test'])
        print("===== TRAIN =======")
        print(r['train'])
        print("===== VALID =======")
        print(r['valid'])

        acc.append(r["test"]["acc"])
        sens.append(r["test"]["sens"])
        spec.append(r["test"]["spec"])
        f1.append(r["test"]["f1"])#'''

    '''
    print("================ FINAL RESULT ==========================")
    print('Acuracia:')
    print(np.mean(acc))
    print(np.std(acc))
    print('Sens:')
    print(np.mean(sens))
    print(np.std(sens))
    print('Spec:')
    print(np.mean(spec))
    print(np.std(spec))
    print('F1:')
    print(np.mean(f1))
    print(np.std(f1))#'''
