import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import re
from lib.datasets import load_datasets
from lib.knn import KnnViewModelling
from lib.gnb import GnbViewModelling
from lib.parzen_window import ParzenViewModelling


"""""""""""""""""""""""""""""""""""""""
Load normalized data and identify views 
for mixture classification 
"""""""""""""""""""""""""""""""""""""""

# load normalized datasets      
X, y = load_datasets()     

# split view data
fac = pd.DataFrame(X[0]); fou = pd.DataFrame(X[1]); kar = pd.DataFrame(X[2])

# rename columns for view identification
fac.columns = ['fac_'+str(i) for i in fac.columns]
fou.columns = ['fou_'+str(i) for i in fou.columns]
kar.columns = ['kar_'+str(i) for i in kar.columns]

# combine data
data = pd.concat([fac, fou, kar, pd.DataFrame({'target_original':y})], axis=1) 



"""""""""""""""""""""""""""""""""""""""
import data and get targets 
from partitions algorithm
"""""""""""""""""""""""""""""""""""""""

# read partition target classification and add data
with open('output/crisp_partitions_all.txt') as f:
    partition_targets = f.readlines()
    for i in range(0,10):
        partition = partition_targets[i][partition_targets[i].find("[")+1:partition_targets[i].find("]")]
        partition = partition.split(', ')
        partition = [int(i) for i in partition]
        data['target_partition'+str(i+1)] = partition


"""""""""""""""""""""""""""""""""""""""
search for K and h parameters
and store accuracy and std dev
"""""""""""""""""""""""""""""""""""""""

def ten_kfold_experiment(dataset, model, times_kfold, parameter):
    ''' execute kfolds experiment of model based in dataset information
        and parameter entry 
    '''
    experiments_results = []
    cont=1
    while cont <= times_kfold:
        # shuffle data to make 10 kfold 
        kdata = dataset.sample(frac=1).reset_index(drop=True)
        kdata['kfold'] = [j for j in range(10) for i in range(200)]
        for fold in kdata['kfold'].unique():
            # select fold test and train data
            train_data = kdata[kdata['kfold']!=fold].reset_index(drop=True)
            test_data = kdata[kdata['kfold']==fold].reset_index(drop=True)
            if 'KNN' in model:
                acc = KnnViewModelling(train_data, test_data, parameter)
            elif 'Parzen' in model:
                acc = ParzenViewModelling(train_data, test_data, parameter)
            elif 'GNB' in model:
                acc = GnbViewModelling(train_data, test_data)
            experiments_results.append(acc)
        cont+=1
    return experiments_results


def searchParameter(dataset, model, times_kfold, parameter_list, get_distribution=False):

    targets =  ['target_original'] + ['target_partition'+str(i) for i in range(1,11)] # in range(1,11)
    experiment_results = {'target':[],'parameter':[], 'accuracy':[],'standard_deviation':[]}

    for target in targets:
        print(target)
        dataset['target'] = dataset[target]
        for parameter in parameter_list:
            print('parameter'+ str(parameter))
            # execute kfold experiment 
            exp_res = ten_kfold_experiment(dataset, model, times_kfold, parameter)
            # get mean and standard deviation
            if get_distribution == False:
                print(np.mean(exp_res))
                experiment_results['accuracy'].append(np.mean(exp_res)) 
                experiment_results['standard_deviation'].append(np.std(exp_res)) 
                experiment_results['parameter'].append(parameter)
                experiment_results['target'].append(target)
            else:
                experiment_results['accuracy'].append(exp_res) 
                experiment_results['parameter'].append(parameter)
                experiment_results['target'].append(target)
    # delete std if search false            
    if get_distribution == True: 
        del experiment_results['standard_deviation']
    return pd.DataFrame(experiment_results)


# search K for KNN
knn_search = searchParameter(data, 'KNN_', times_kfold = 30, parameter_list = range(1,51))
knn_search.to_csv('output/searches/knn_search_k.csv', index=False)

# search h for Parzen Window 
parzen_search = searchParameter(data, 'Parzen_', times_kfold = 30, parameter_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
parzen_search.to_csv('output/searches/parzen_search_h.csv', index=False)



"""""""""""""""""""""""""""""""""""""""""""""""""""
visualize search parameters
"""""""""""""""""""""""""""""""""""""""""""""""""""

def plotSearch(df):
    df = df[df.target == 'target_partition2']
    print(df[df.accuracy == max(df.accuracy)])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.errorbar(df.parameter, df.accuracy,
                #yerr=df.standard_deviation,
                fmt='-o')
    ax.set_xlabel('Parâmetro K')
    ax.set_ylabel('Acurácia / Desvio-padrão')
    plt.show()


plotSearch(knn_search)
plotSearch(parzen_search)

"""""""""""""""""""""""""""""""""""""""""""""""""""
execute experiments with best accuracy parameter
for KNN and Parzen Window
"""""""""""""""""""""""""""""""""""""""""""""""""""

# GNB
gnb = searchParameter(data, 'GNB_', times_kfold = 30, parameter_list = ['dummy'], get_distribution=True)
gnb.to_csv('output/results_gnb.csv', index=False)

# KNN
knn = searchParameter(data, 'KNN_', times_kfold = 30, parameter_list = [23], get_distribution=True)
knn.to_csv('output/results_knn.csv', index=False)

# Parzen Window
parzen = searchParameter(data, 'Parzen_', times_kfold = 30, parameter_list = [0.4], get_distribution=True)
parzen.to_csv('output/results_pw.csv', index=False)


"""""""""""""""""""""""""""""""""""""""""""""""""""
EVALUATION METRICS
"""""""""""""""""""""""""""""""""""""""""""""""""""


# import experiment results
gnb = pd.read_csv('output/results_gnb.csv')
knn = pd.read_csv('output/results_knn.csv')
parzen = pd.read_csv('output/results_pw.csv')

# name models and combine
gnb['model'] = 'gnb'
knn['model'] = 'knn'
parzen['model'] = 'parzen'
models = pd.concat([gnb, knn, parzen]).reset_index(drop=True)


def metricsTable(models, target_partition, version=''):

    # accuracy and confidence interval of best models
    best_models = models[models.target == target_partition].reset_index(drop=True) 
    table1 = {'model':[],'mean_accuracy':[],'confidence_interval_90':[], 'confidence_interval_95':[]}

    for i in range(len(best_models)):
        best_models.accuracy[i] = best_models.accuracy[i].replace(']','').replace('[','').split(', ')
        best_models.accuracy[i] = [float(x) for x in best_models.accuracy[i]]
        acc = best_models.accuracy[i]
        table1['model'].append(best_models.model[i]  + '_' + best_models.target[i]) 
        table1['mean_accuracy'].append( np.mean(acc) ) 
        table1['confidence_interval_90'].append( (np.percentile(acc, 5.0), np.percentile(acc, 95.0))   ) 
        table1['confidence_interval_95'].append( (np.percentile(acc, 2.5), np.percentile(acc, 97.5))   ) 

    table1 = pd.DataFrame(table1)
    table1.to_csv('output/results_main2'+version+'.csv')

    return best_models


# best partition on part 1
models_test = metricsTable(models, 'target_partition2')

# original target for comparison
metricsTable(models, 'target_original',version='_original')

# friedman test
from scipy.stats import friedmanchisquare
stat, p = friedmanchisquare(models_test.accuracy[0], models_test.accuracy[1], models_test.accuracy[2])
print('Friedman Test for All=%.6f, p=%.20f' % (stat, p))

# paired wilcoxon
from scipy.stats import wilcoxon

stat, p = wilcoxon(models_test.accuracy[0], models_test.accuracy[1])
print('Wilcoxon Test GNB/KNN =%.6f, p=%.6f' % (stat, p))

stat, p = wilcoxon(models_test.accuracy[0], models_test.accuracy[2])
print('Wilcoxon Test GNB/Parzen =%.6f, p=%.6f' % (stat, p))

stat, p = wilcoxon(models_test.accuracy[1], models_test.accuracy[2])
print('Wilcoxon Test KNN/Parzen =%.6f, p=%.6f' % (stat, p))