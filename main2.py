import pandas as pd
import numpy as np
import re
from lib.datasets import load_datasets
from lib.knn import KnnViewModelling
#from lib.parzen_window import *

"""""""""""""""""""""""""""""""""""""""
Load normalized data, combine in df
and shuffle for k-fold
"""""""""""""""""""""""""""""""""""""""

# load normalized datasets      
X, y = load_datasets()     

# split data
fac = pd.DataFrame(X[0]); fou = pd.DataFrame(X[1]); kar = pd.DataFrame(X[2])

# rename columns
fac.columns = ['fac_'+str(i) for i in fac.columns]
fou.columns = ['fou_'+str(i) for i in fou.columns]
kar.columns = ['kar_'+str(i) for i in kar.columns]

# combine data
data = pd.concat([fac, fou, kar, pd.DataFrame({'target_original':y})], axis=1) 

# add partition target classification
with open('output/crisp_partitions_all.txt') as f:
    partition_targets = f.readlines()
    for i in range(0,10):
        partition = partition_targets[i][partition_targets[i].find("[")+1:partition_targets[i].find("]")]
        partition = partition.split(', ')
        data['target_partition'+str(i+1)] = partition

# random shuffle data and make 10 k-fold
data = data.sample(frac=1).reset_index(drop=True)
data['kfold'] = np.array([j for j in range(10) for i in range(200)])

"""""""""""""""""""""""""""""""""""""""
search for K and h parameters
"""""""""""""""""""""""""""""""""""""""

def kfold_experiment(dataset, model, kfolds, parameter):
    ''' execute kfolds experiment of model based in dataset information
        and parameter entry 
    '''
    experiments_results = []
    for fold in dataset['kfold'].unique():
        cont=1
        while cont <= kfolds:
            # select fold test and train data
            train_data = dataset[dataset['kfold']!=fold].reset_index(drop=True)
            test_data = dataset[dataset['kfold']==fold].reset_index(drop=True)
            if 'KNN' in model:
                # KNN model
                acc = KnnViewModelling(train_data, test_data, parameter)
            else:
                acc = parzen_view_modelling(train_data, test_data)
            # append results
            experiments_results.append(acc)
            cont+=1
    return pd.DataFrame({'results_'+model: experiments_results})


def searchParameter(dataset, model, kfolds, parameter_list):
    targets =  ['target_original'] + ['target_partition'+str(i) for i in range(1,11)]
    experiment_results = {'target':[],'parameter':[], 'accuracy':[],'standard_deviation':[]}
    for target in targets:
        dataset['target'] = dataset[target]
        for parameter in parameter_list:
            print('searching for '+ str(parameter))
            # execute kfold experiment 
            exp_res = kfold_experiment(dataset, model+str(parameter), kfolds, parameter)
            # get mean and standard deviation
            experiment_results['accuracy'].append(np.mean(exp_res)) 
            experiment_results['standard_deviation'].append(np.std(exp_res)) 
            experiment_results['parameter'].append(parameter)
            experiment_results['target'].append(target)
    return pd.DataFrame(experiment_results)


# search K in original target
knn_search = searchParameter(data, 'KNN_', 10, range(1,51))
knn_search.to_csv('output/searches/KNN_search_k_full.csv')

# search K in Q-Partitons target
