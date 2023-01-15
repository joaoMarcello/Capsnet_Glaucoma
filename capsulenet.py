"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

from keras import layers, models, optimizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

from joblib import Parallel, delayed
import multiprocessing as mp
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import pickle

# ------------------- Setting the seeds from reproducibility -----------------------
import os
os.environ['PYTHONHASHSEED'] = '0'

import random
from random import seed, randint
seed(123)

import numpy as np
np.random.seed(42)
# ---------------------------------------------------------------------------------

# ---------------------------- Configuring gpu ------------------------------------------
import tensorflow as tf
session_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
session_config.gpu_options.allow_growth = True

from keras import backend as K

tf.set_random_seed(123)
#K.set_session(tf.Session(graph=tf.get_default_graph(), config=session_config))
#----------------------------------------------------------------------------------------


K.set_image_data_format('channels_last')


from dataGenerator import *
from dataset_utils import *
from augmentation import *

import cv2


# ======================== MY CUTOM METRICS ================================================
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
# =====================================================================================================================
def customMetricsNP(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(np.argmax(y_true, 1), np.argmax(y_pred, 1)).ravel()

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
# ======================================================================================================================

def CapsNet(input_shape, n_class, routings, args):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    inputs = layers.Input(shape=input_shape, name='input_layer')

    # Layer 1: Just a conventional Conv2D layer
    x = layers.Conv2D(filters=int(256*args['filters_mult']), kernel_size=(args['kernel_size'],args['kernel_size']), strides=1, padding='valid', activation=args['activation'], name='conv1')(inputs)
    x = layers.Conv2D(filters=int(256*args['filters_mult']), kernel_size=(args['kernel_size'],args['kernel_size']), strides=1, padding='valid', activation=args['activation'], name='conv2')(x)
    x = layers.Conv2D(filters=int(256*args['filters_mult']), kernel_size=(args['kernel_size'],args['kernel_size']), strides=1, padding='valid', activation=args['activation'], name='conv3')(x)
    #x = layers.Conv2D(filters=int(256*args['filters_mult']), kernel_size=(5,5), strides=1, padding='valid', activation=args['activation'], name='conv4')(x)


    inputs2 = layers.Input(shape=(112,112, 1), name = 'input_layer_vessel')
    m = layers.Conv2D(filters=int(256*args['filters_mult']), kernel_size=(args['kernel_size'],args['kernel_size']), strides=1, padding='valid', activation=args['activation'], name='vessel_conv1')(inputs2)
    m = layers.Conv2D(filters=int(256*args['filters_mult']), kernel_size=(args['kernel_size'],args['kernel_size']), strides=1, padding='valid', activation=args['activation'], name='vessel_conv2')(m)
    m = layers.Conv2D(filters=int(256*args['filters_mult']), kernel_size=(args['kernel_size'],args['kernel_size']), strides=1, padding='valid', activation=args['activation'], name='vessel_conv3')(m)


    combined = layers.concatenate([x, m])

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(combined, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([inputs, inputs2, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model( [inputs, inputs2], [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([inputs, inputs2, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train, vessel_train), (x_test, y_test, vessel_valid) = data

    # callbacks
    from keras.callbacks import CSVLogger
    from keras.callbacks import TensorBoard
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import LearningRateScheduler
    from keras.callbacks import EarlyStopping
    from keras.preprocessing.image import ImageDataGenerator

    log = CSVLogger(args['save_dir'] + '/log.csv')
    tb = TensorBoard(log_dir=args['save_dir'] + '/tensorboard-logs',
                               batch_size=args['batch_size'], histogram_freq=int(False)) #args['debug']
    checkpoint = ModelCheckpoint(args['save_dir'] + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = LearningRateScheduler(schedule=lambda epoch: args['lr'] * (args['lr_decay'] ** epoch))

    early = EarlyStopping(monitor='val_capsnet_acc', min_delta=0.03, patience=int(0.7 * args['epochs']), verbose=0, mode='max')
    
    # compile the model
    opt = optimizers.Adam(lr=args['lr'])
    if args['optimizer'] == 'sgd': opt = optimizers.SGD(lr=args['lr'])
    else:
        if args['optimizer'] == 'adadelta': opt = optimizers.Adadelta(lr=args['lr'])
        else:
            if args['optimizer'] == 'adagrad': opt = optimizers.Adagrad(lr=args['lr'])
            else:
                if args['optimizer'] == 'adamax': opt = optimizers.Adamax(lr=args['lr'])
                else:
                    if args['optimizer'] == 'nadam': opt = optimizers.Nadam(lr=args['lr'])

    model.compile(optimizer=opt,
                  loss=[margin_loss, args['loss']],
                  loss_weights=[1., args['lam_recon']],
                  metrics={'capsnet': ['accuracy', Acc, Sens, Spec, F1]})

    
    #======================= My training ===================================================================

    paramTrainGen = {
        'dim' : (112,112), # tamanho desejado para as imagens
        'batch_size' : args['batch_size'], # tamanho do batch
        'n_classes' : 2,   # num. de  classes
        'n_channels' : 3,  # num. de canais
        'shuffle' : True,   # se os elementos do batch sao gerados randomicamente
        'data' : x_train,
        'vessel' : vessel_train
    }
    paramTestGen = {
        'dim' : (112,112), # tamanho desejado para as imagens
        'batch_size' : args['batch_size'], # tamanho do batch
        'n_classes' : 2,   # num. de  classes
        'n_channels' : 3,  # num. de canais
        'shuffle' : True,   # se os elementos do batch sao gerados randomicamente
        'data' : x_test,
        'vessel' : vessel_valid
    }

    training_generator = DataGenerator(np.asarray(range(len(x_train))), np.argmax(y_train, 1), **paramTrainGen)
    valid_generator = DataGenerator(np.asarray(range(len(x_test))), np.argmax(y_test, 1), **paramTestGen)

    print('+====================================== GOING TO TRAIN ==========================================')
    print('|    Train: ' + str(len(x_train)))
    print(np.bincount(np.argmax(y_train, 1)))
    print('|    Valid: ' + str(len(x_test)))
    print(np.bincount(np.argmax(y_test, 1)))
    print('+================================================================================================')

    history = model.fit_generator(generator=training_generator, 
                                steps_per_epoch = (y_train.shape[0] / args['batch_size']), 
                                validation_data= valid_generator,
                                validation_steps = (y_test.shape[0] / paramTestGen['batch_size']),
                                epochs = args['epochs'],
                                callbacks = [log, tb, lr_decay, early])
    #=======================================================================================================#'''

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args['batch_size'], epochs=args['epochs'],
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay, early])
    #"""



    """
    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    trained_model = model.fit_generator(generator=train_generator(x_train, y_train, args['batch_size'], args['shift_fraction']),
                        steps_per_epoch=int(y_train.shape[0] / args['batch_size']),
                        epochs=args['epochs'],
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay, early])
    # End: Training with data augmentation -----------------------------------------------------------------------#
    #"""


    # model.save_weights(args['save_dir'] + '/trained_model.h5')
    # print('Trained model saved to \'%s/trained_model.h5\'' % args['save_dir'])



    #saveModel(model, args['save_dir'])

    from utils import plot_log
    #plot_log(args['save_dir'] + '/log.csv', show=True)

    return model, history

def saveModel(model, path, modelName='model_final'):
    # serialize model to JSON
    model_json = model.to_json()
    with open(path + modelName + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path + modelName + ".h5")


def test(model, data, batch_size= None):
    from keras.utils import to_categorical

    x_test, y_test, vessel_test = data
    y_pred, x_recon = model.predict([x_test, vessel_test], batch_size=batch_size) #, batch_size=args['batch_size']

    dic = customMetricsNP(y_test, y_pred)
    print('======================================================')
    print(dic)
    print('======================================================')

    return np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0], dic
    '''
    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args['save_dir'] + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args['save_dir'])
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args['save_dir']+ "/real_and_recon.png"))
    plt.show()#'''


def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args['digit']
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args['save_dir'] + '/manipulate-%d.png' % args['digit'])
    print('manipulated result saved to %s/manipulate-%d.png' % (args['save_dir'], args['digit']))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)




def calculateMean(x):
    soma = np.array([0., 0., 0.])

    for i, img in enumerate(x):
        r = np.sum(img, (0,1))
        soma = soma + r
    
    return soma/(x.shape[0] * x[0].shape[0] * x[0].shape[0])





def histogram_matching(data, reference):
    from skimage.exposure import match_histograms

    x = []

    for img in data:
        matched = match_histograms(img, reference, multichannel=True)
        x.append(matched)

    return np.asarray(x).astype('float32')


def objective(space):
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    '''
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()#'''


    if not os.path.exists(space['save_dir']):
        os.makedirs(space['save_dir'])

    from crossValidationData import loadMeanMinMax, preprocess


    x_train = load_hdf5('./CrossValidData/crossvalid_train_' + space['folder_number'] + '_dataset.hdf5')
    y_train = load_hdf5('./CrossValidData/crossValid_train_' + space['folder_number'] + '_labels.hdf5')
    vessel_train = load_hdf5('./CrossValidData/vessel_train_' + space['folder_number'] + '.hdf5')

    x_valid = load_hdf5('./CrossValidData/crossvalid_valid_' + space['folder_number'] + '_dataset.hdf5')
    y_valid = load_hdf5('./CrossValidData/crossValid_valid_' + space['folder_number'] + '_labels.hdf5')
    vessel_valid = load_hdf5('./CrossValidData/vessel_valid_' + space['folder_number'] + '.hdf5')

    x_test = load_hdf5('./test_dataset.hdf5')
    y_test= load_hdf5('./test_labels.hdf5')
    vessel_test = load_hdf5('./vessel/vessel_test_dataset.hdf5')

    mean, minVal, maxVal = loadMeanMinMax('./CrossValidData/', 'mean-min-max_train_' + space['folder_number'])
    x_test, mean, minVal, maxVal = preprocess(x_test, mean, minVal, maxVal)

    args = space['space']
    args['save_dir'] = space['save_dir']


    print('before: ', x_train.shape, x_valid.shape)
    x_train, y_train, vessel_train = balancedDataAugmentation(x_train, y_train, vessel_train)
    print('after: ', x_train.shape, x_valid.shape)

    
    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=2,
                                                  routings=args['routings'], args=args)
    #eval_model.summary()#'''


    '''r, dic =test(eval_model, (x_train, y_train, vessel_train))
    r, dic =test(eval_model, (x_test, y_test, vessel_test))
    r, dic =test(eval_model, (x_valid, y_valid, vessel_valid))

    print(np.bincount(np.argmax(y_train, 1)))#'''


    #print(r[0])
    #r = model.predict( [np.zeros((1, 112, 112, 3)), np.zeros((1, 112, 112, 1)), np.zeros( (1,2))], batch_size=None )

    
    try:
        model, history = train(model=model, data=((x_train, y_train, vessel_train), (x_valid, y_valid, vessel_valid)), args=args)
    except:
        return -1#'''

    
    #model.load_weights('D:/JoaoMarcello/Documents/CNPQ/Program/cap-glaucoma-12-11/result/weights-01.h5')



    resultTest, metricsTest = test(model=eval_model, data=(x_test, y_test, vessel_test))

    resultValid, metricsValid = test(model=eval_model, data=(x_valid, y_valid, vessel_valid))

    resultTrain, metricsTrain = test(model=eval_model, data=(x_train, y_train, vessel_train))


    with open(args['save_dir']+'history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    if metricsValid['sens'] != 0. and metricsValid['spec'] != 0. and metricsTest['sens'] != 0. and metricsTest['spec'] != 0.: 
        saveModel(model, args['save_dir'], 'final_model')


    return {
        "train" : metricsTrain,
        "valid" : metricsValid,
        "test" : metricsTest
    }#'''
