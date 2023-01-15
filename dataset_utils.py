import h5py
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split


def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
    return f["image"][()]

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)
    

def loadImages(pathToDataset, x, y, data, labels, allPaths, size=(112, 112), mode=1):
    from glob import glob

    pathsGlaucoma = glob(pathToDataset + 'Glaucoma/*')
    pathsHealthy = glob(pathToDataset + 'Non-Glaucoma/*')

    for paths in pathsGlaucoma:
        img = cv2.imread(paths, mode)
        img = cv2.resize(img, size)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data[ paths[len(pathToDataset):] ] = img
        labels[paths[len(pathToDataset):]] = 1
        x.append(img)
        y.append(1.0)

    for paths in pathsHealthy:
        img = cv2.imread(paths, mode)
        img = cv2.resize(img, size)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data[paths[len(pathToDataset):]] = img
        labels[paths[len(pathToDataset):]] = 0
        x.append(img)
        y.append(0.0)

    return x, y, data, labels, allPaths + [p[len(pathToDataset):] for p in pathsGlaucoma] + [p[len(pathToDataset):] for p in pathsHealthy]

def getPaths(pathToDataset, name=''):
    from glob import glob

    pathsGlaucoma = glob(pathToDataset + 'Glaucoma/*')
    pathsHealthy = glob(pathToDataset + 'Non-Glaucoma/*')

    return [ name + '/' + p[len(pathToDataset):] for p in pathsGlaucoma] + [name + '/' + p[len(pathToDataset):] for p in pathsHealthy]

def getAllPaths(pathToDataset, names=['Acrima', 'Dristhi', 'Hrf',  'iChallenge-Gon', 'Jsiec', 'Rim-one-1', 'Rim-one-2', 'Rim-one-3']):
    p = []
    for n in names:
        p = p + getPaths(pathToDataset + '/' + n + '/', name=n)

    return p

def process_clahe(lista):
    x = []

    for i, imagem in enumerate(lista):

        imagem = cv2.normalize(imagem, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

        imagem = cv2.fastNlMeansDenoisingColored(imagem,None,10,10,7,21)

        lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)

        lab_planes = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv2.merge(lab_planes)

        imagem = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        x.append(imagem)
        #s = pathToWrite + lista[i][-9:]#'{:04d}'.format(i) + ".jpg"

        #cv2.imwrite(s, imagem)
        print("Imagem", i+1)


    x = np.asarray(x)
    return x

def load_glaucoma_dataset(pathToDataset, mode=1, names=['Acrima', 'Dristhi', 'Hrf',  'iChallenge-Gon', 'Jsiec', 'Rim-one-1', 'Rim-one-2', 'Rim-one-3']):
    import cv2
    from keras.utils import to_categorical 
    from sklearn.model_selection import train_test_split

    x = []
    y = []
    data = {}
    labels = {}
    paths = []

    for n in names:
        x, y, data, labels, paths = loadImages(pathToDataset + n + '/', x, y, data, labels, paths, mode=mode)

    x = np.array(x, dtype='float32')

    y = np.asarray(y).astype('float32')

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    #x_train = x_train.reshape(-1, 112, 112, 3).astype('float32') / 255.
    #x_test = x_test.reshape(-1, 112, 112, 3).astype('float32') / 255.
    #y_train = to_categorical(y_train.astype('float32'))
    
    y = to_categorical(y, 2)

    return x, y, paths

def saveProcessedDatasetHdf5(x, y, name='', pathToWrite='./Processed_dataset'):
    if not os.path.exists(pathToWrite):
        os.makedirs(pathToWrite)

    write_hdf5(x, pathToWrite + "/" + name + "_dataset.hdf5")
    write_hdf5(y, pathToWrite + "/" + name + "_labels.hdf5")

def saveProcessedDataset(pathToDataset, names):
    for n in names:
        x, y, data, labels, paths = loadImages(pathToDataset + n + '/', [], [], {}, {}, [])
        x = process_clahe(x)

        for c, img in enumerate(x):
            savePath = './Processed_dataset/' + n + '/'
            if not os.path.exists(savePath + 'Glaucoma/'):
                os.makedirs(savePath + 'Glaucoma/')
            if not os.path.exists(savePath + 'Non-Glaucoma/'):
                os.makedirs(savePath + 'Non-Glaucoma/')

            cv2.imwrite(savePath + paths[c], img)

        print(x.shape)

def separateDataset(pathToWrite='./', percent=0.1, mode=1):
    names = ['Acrima', 'Dristhi', 'Hrf',  'iChallenge-Gon', 'Jsiec', 'Rim-one-1', 'Rim-one-2', 'Rim-one-3']
    pathToDataset = 'D:/JoaoMarcello/Documents/Datasets/Dataset_2020/Dataset-vessel/'
    #pathToDataset = 'D:/JoaoMarcello/Documents/CNPQ/Program/2020/project_dataset/Processed_dataset/'

    if not os.path.exists(pathToWrite): os.makedirs(pathToWrite)

    for n in names:
        x, y, data, labels, paths = loadImages(pathToDataset + n + '/', [], [], {}, {}, [], mode=mode)

        x = np.asarray(x)
        
        x_train, x_test, y_train, y_test = train_test_split(paths, y, test_size = percent, random_state = 42)

        print(n+':   Train:', np.bincount(y_train), '   Test:',np.bincount(y_test))

        if not os.path.exists(pathToWrite + 'Train_dataset/' + n + '/Glaucoma/'): os.makedirs(pathToWrite + 'Train_dataset/' + n + '/Glaucoma/')
        if not os.path.exists(pathToWrite + 'Train_dataset/' + n + '/Non-Glaucoma/'): os.makedirs(pathToWrite + 'Train_dataset/' + n + '/Non-Glaucoma/')

        for c, p in enumerate(x_train):
            img = data[p]
            if y_train[c] == 0:
                cv2.imwrite(pathToWrite + 'Train_dataset/' + n + '/' + p, img)
            else:
                cv2.imwrite(pathToWrite + 'Train_dataset/' + n + '/' + p, img)


        if not os.path.exists(pathToWrite + 'Test_dataset/' + n + '/Glaucoma/'): os.makedirs(pathToWrite + 'Test_dataset/' + n + '/Glaucoma/')
        if not os.path.exists(pathToWrite + 'Test_dataset/' + n + '/Non-Glaucoma/'): os.makedirs(pathToWrite + 'Test_dataset/' + n + '/Non-Glaucoma/')

        for c, p in enumerate(x_test):
            img = data[p]
            if y_test[c] == 0:
                cv2.imwrite(pathToWrite + 'Test_dataset/' + n + '/' + p, img)
            else:
                cv2.imwrite(pathToWrite + 'Test_dataset/' + n + '/' + p, img)


def load_save_vessel():
    def turn_img_bin(img, tres=0.5, maxValue=1.):
        t, img = cv2.threshold(img, tres, maxValue, cv2.THRESH_BINARY)
        return np.expand_dims(img, axis=2)

    x, y, paths = load_glaucoma_dataset('./vessel/Test_dataset/', mode=0)  
    print(x.shape, np.bincount(np.argmax(y,1)))

    x = np.expand_dims(x/255., axis=3)
    x = [turn_img_bin(img) for img in x]
    x = np.array(x, 'float32')

    print(x.shape, np.bincount(np.argmax(y,1)))

    saveProcessedDatasetHdf5(x, y, 'vessel_test', './vessel/')#'''



if __name__ == '__main__':
    
    '''x, y, paths = load_glaucoma_dataset('./vessel/Test_dataset/', mode=0)  
    print(x.shape, np.bincount(np.argmax(y,1)))

    saveProcessedDatasetHdf5(x, y, 'vessel', './vessel/')#'''

    #separateDataset('./vessel/', mode=0)

    load_save_vessel()