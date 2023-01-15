import gc
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pickle
import time
import numpy as np


trials = pickle.load(open("./result/" + "teste_16-03" + "_trials.p", "rb"))

print(len(trials.trials))
print(trials.best_trial['result']['space'])
for c, t in enumerate(trials.trials):
    try:
        print(t['result']['metrics']['acc_mean'], t['result']['metrics']['sens_mean'], t['result']['metrics']['spec_mean'], t['result']['metrics']['f1_mean'] )
    except:
        print()


# TESTES COM SEGMENTAÇÂO DOS VASOS

'''
for i, t in enumerate(trials.trials):
    try:
        
        if t['result']['args']['loss'] == 'categorical_crossentropy':
            print('\n', t['result']['args'])
            #print(i,  t['result']['result']['acc'],  t['result']['result']['sens'],  t['result']['result']['spec'], t['result']['args']['loss'] )
    except:
        a = 1#'''


'''
20
{'loss': -0.8348623853211009, 'status': 'ok', 
'args': 
{'activation': 'elu', 
    'batch_size': 30, 
    'epochs': 75,
    'filters_mult': 0.7921006985981938, 
    'kernel_size': 5, 
    'lam_recon': 0.2038285118295179, 
    'loss': 'mean_squared_logarithmic_error', 
    'lr': 0.0001773204794692644, 
    'lr_decay': 0.5782598954211678, 
    'optimizer': 'adagrad', 
    'routings': 3, 
    'save_dir': 'result/model_13/'}, 

'result': {'tp': 50, 'tn': 132, 'fp': 21, 'fn': 15, 'sens': 0.7692307692307693, 'spec': 0.8627450980392157, 'acc': 0.8348623853211009, 'prec': 0.704225352112676, 'f1': 0.7352941176470589}, 'eval_time': 1579021725.5197268}
'''