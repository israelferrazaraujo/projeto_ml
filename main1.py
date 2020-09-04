from lib.datasets import load_datasets, dissimilarity_matrices
from lib.metrics import calc_metric
from lib.MFCMdd import MFCMdd_RWL_P


"""""""""""""""""""""""""""""""""""""""
Load the normalized data sets and 
calculate the dissimilarity using 
Euclidean distance
"""""""""""""""""""""""""""""""""""""""


X, y = load_datasets()                # load normalized datasets
D = dissimilarity_matrices(X)         # dissimilarity using euclidian distance


"""""""""""""""""""""""""""""""""""""""
Calculates fuzzy partitions and prototypes
for each parameter "q" on q_list, using
all dissimilarity matrices simultaneously
and individually
"""""""""""""""""""""""""""""""""""""""



# list of "q" parameters
q_list = list(range(1, 11))

results = {'all': {}, 'fac': {}, 'kar': {}, 'fou': {}}

# all dissimilarity matrices simultaneously
results['all'] = dict((q, MFCMdd_RWL_P(D, q=q, reps=100)) 
                        for q in q_list) # dict. to store the results (U, G)

# all dissimilarity matrices individually
for i, mfeat in enumerate(list(results.keys())[1:]):
    results[mfeat] = dict((q, MFCMdd_RWL_P(D[i:i+1], q=q, reps=100))
                        for q in q_list) 



"""""""""""""""""""""""""""""""""""""""
Calculates metrics for each "q"
"""""""""""""""""""""""""""""""""""""""



metrics = dict((mfeat, 
                dict((q, calc_metric(result[0], result[1], y)) 
                    for q, result in mfeat_results.items())) 
              for mfeat, mfeat_results in results.items())



"""""""""""""""""""""""""""""""""""""""
Stores crisp partitions in one file
"""""""""""""""""""""""""""""""""""""""


for mfeat, mfeat_metrics in metrics.items():
    with open('output/crisp_partitions_{}.txt'.format(mfeat), 'w') as f:
        for q, metric in mfeat_metrics.items():
            print('q={} : {}'.format(q, metric['crisp_partition']), file=f)


"""""""""""""""""""""""""""""""""""""""
Stores all metrics in one file
"""""""""""""""""""""""""""""""""""""""


for mfeat, mfeat_metrics in metrics.items():
    with open('output/results_{}.txt'.format(mfeat), 'w') as f:
        for q, metric in mfeat_metrics.items():
            print('q={}:'.format(q), file=f)
            for key, value in metric.items():
                print(key,'\n', value, file=f)

        
