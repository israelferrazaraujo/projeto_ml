

# import paackages
import pandas as pd
import numpy as np

# import experiment results
gnb = pd.read_csv('output/searches/GNB.csv')
knn = pd.read_csv('output/searches/knn.csv')
parzen = pd.read_csv('output/searches/parzen.csv')

# name models and combine
gnb['model'] = 'gnb'
knn['model'] = 'knn'
parzen['model'] = 'parzen'
models = pd.concat([gnb, knn, parzen]).reset_index(drop=True)


# accuracy and confidence interval of best models
best_models = models[models.target == 'target_partition2'].reset_index(drop=True) 
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
table1.to_csv('output/results_part2.csv')

# friedman 
from scipy.stats import friedmanchisquare
stat, p = friedmanchisquare(best_models.accuracy[0], best_models.accuracy[1], best_models.accuracy[2])
print('Friedman Test for All=%.3f, p=%.3f' % (stat, p))

# paired wilcoxon
from scipy.stats import wilcoxon

stat, p = wilcoxon(best_models.accuracy[0], best_models.accuracy[1])
print('Wilcoxon Test GNB/KNN =%.3f, p=%.3f' % (stat, p))

stat, p = wilcoxon(best_models.accuracy[0], best_models.accuracy[2])
print('Wilcoxon Test GNB/Parzen =%.3f, p=%.3f' % (stat, p))

stat, p = wilcoxon(best_models.accuracy[1], best_models.accuracy[2])
print('Wilcoxon Test KNN/Parzen =%.3f, p=%.3f' % (stat, p))