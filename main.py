import gc
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import pickle
import time
import numpy as np
from keras.callbacks import CSVLogger
from capsulenet import *

from joblib import Parallel, delayed
import multiprocessing as mp
import time

from dataset_utils import *

# Classification Report sklearn

space = {
    'batch_size' : hp.choice('batch_size', np.arange(15, 35, 5, dtype = int) ),
    'lr' : hp.loguniform('lr', -9.90348, -6.90775),
    'lr_decay' : hp.loguniform('lr_decay', -2.30258, -0.10536),
    'lam_recon' : hp.uniform('lam_recon', 0.1, 0.4),
    'routings' : 3,
    #'shift_fraction' : 0.1,
    #'debug' : False,
    #'save_dir' : './result',
    #'testing' : False,
    #'digit' : 5,
    #'weights' : None,
    'optimizer' : hp.choice('optimizer', ['sgd', 'adadelta', 'adagrad', 'adam', 'adamax', 'nadam']),
    'activation' : hp.choice('activation', ['relu', 'tanh', 'elu']),
    #'dropout' : hp.uniform('dropout', 0.1, 0.5),
    #'kernel_constraint' : hp.choice('kernel_constraints', [3, 4]),
    #'units_mult' : hp.uniform('units_mult', 0.3, 1.0),
    'epochs' : hp.choice('epochs',  np.arange(40, 65, 5, dtype = int)),
    #'num_units' : hp.choice('num_units', [2,3,4]),
    'kernel_size' : hp.choice('kernel_size', [5,7,9,11]),
    'filters_mult' : hp.uniform('filters_mult', 0.5, 1.2),
    'loss' : hp.choice('loss', ['categorical_crossentropy', 'mean_squared_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'categorical_hinge'])
}

# Classification Report sklearn

'''
space = {
    'batch_size' : 25,
    'lr' : 0.0008158114894119092,
    'lr_decay' : 0.8838238952823108,
    'lam_recon' : 0.2116793651298763,
    'routings' : 3,
    'shift_fraction' : 0.1,
    'debug' : False,
    'save_dir' : './result',
    'testing' : False,
    'digit' : 5,
    'weights' : None,
    'optimizer' : 'adadelta',
    'activation' : 'elu',
    #'dropout' : hp.uniform('dropout', 0.1, 0.5),
    #'kernel_constraint' : hp.choice('kernel_constraints', [3, 4]),
    #'units_mult' : hp.uniform('units_mult', 0.3, 1.0),
    'epochs' : 4,
    #'num_units' : hp.choice('num_units', [2,3,4]),
    'kernel_size' : 5,
    'filters_mult' : 0.5110177848157855,
    'loss' : 'squared_hinge'
}#'''


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

    

    for i in range(5):
        s = str(i)
        print('============================== MODEL ' + str(model_number) + ' - TESTE ' + s + ' =======================================')

        args["space"] = space
        args["save_dir"] = './result/modelo_' + str(model_number) + '/teste_' + s + '/'
        args['folder_number'] = s

        if not os.path.exists('./result/modelo_' + str(model_number) + '/'): os.makedirs('./result/modelo_' + str(model_number) + '/')
        
        print('=================== SPACE ===============================')
        print(space)



        # training the model in a separated process and getting the results
        pool = mp.Pool(1)
        r = pool.map(objective, [args] )

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

    def save_metrics(dic, name):
        f = open('./result/modelo_' + str(model_number) + '/' + name + '.txt', 'w')
        v = ['acc_mean', 'acc_std', 'acc_max', 'acc_min', 'sens_mean', 'sens_std', 'sens_max', 'sens_min',
        'spec_mean', 'spec_std', 'spec_max', 'spec_min', 'f1_mean', 'f1_std', 'f1_max', 'f1_min' ]
        [ f.write(actual + ':' + str(dic[actual]) + '\n' ) for actual in v]
        f.close()

    save_metrics(train_metrics, 'train_metrics')
    save_metrics(valid_metrics, 'valid_metrics')
    save_metrics(test_metrics, 'test_metrics')

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



if __name__ == '__main__':
    '''args = {
        "x" : x,
        "y" : y,
        'space' : space
    }

    current = 1

    from sklearn.model_selection import StratifiedKFold, KFold

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # training the model using cross validation
    for train, valid in kfold.split(x, np.argmax(y, 1)):

        print(space)

        args['folder_number'] = '0'

        args['save_dir'] = 'result/model_' + str(n_model) + '/m_' + str(current) + '/' 

        result = []
        pool = mp.Pool(1)
        r = pool.map(objective, [args] )

        break#'''

    '''
    reference = getReference()

    x = histogram_matching(x, reference)
    x = x/255.#'''

    if not os.path.exists('./result/'): os.makedirs('./result/')

    for i in range(40):
        try:
            trials = pickle.load(open("./result/" + "teste_10-06" + "_trials.p", "rb"))
            max_evals = len(trials.trials) + 2
            n_model = len(trials.trials)
            print('--------------- LOAD TRIALS ------------------------')
        except FileNotFoundError:
            trials = Trials()
            max_evals = 2
            n_model = 0

        best = fmin(objective2,
        space=space,
        algo=tpe.suggest,
        max_evals= max_evals,
        trials=trials)  #rstate= np.random.RandomState(400 * i)

        pickle.dump(trials, open("./result/" + "teste_10-06" + "_trials.p", "wb"))#'''


