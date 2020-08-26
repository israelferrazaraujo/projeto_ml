from lib.datasets import load_datasets, dissimilarity_matrices
from lib.metrics import calc_metric
from lib.part_1 import MFCMdd_RWL_P


"""""""""""""""""""""""""""""""""""""""
Load the normalized data sets and 
calculate the dissimilarity using 
Euclidean distance
"""""""""""""""""""""""""""""""""""""""


X, y = load_datasets()                # load normalized datasets
D = dissimilarity_matrices(X)         # dissimilarity using euclidian distance


"""""""""""""""""""""""""""""""""""""""
Calculates fuzzy partitions and prototypes
for each parameter "q" on q_list
"""""""""""""""""""""""""""""""""""""""


q_list = list(range(1, 11))               # list of "q" parameters
results = dict((q, None) for q in q_list) # dict. to store the results (U, G)


for q in q_list:
    results[q] = MFCMdd_RWL_P(D, q=q, reps=100) # returns (U, G)



"""""""""""""""""""""""""""""""""""""""
Calculates metrics for each "q"
"""""""""""""""""""""""""""""""""""""""


metrics = dict((q, calc_metric(result[0], result[1], y)) for q, result in results.items())



"""""""""""""""""""""""""""""""""""""""
Stores crisp partitions in one file
"""""""""""""""""""""""""""""""""""""""


with open('output/crisp_partitions.txt', 'w') as f:
    for q, metric in metrics.items():
        print('q={} : {}'.format(q, metric['crisp_partition']), file=f)


"""""""""""""""""""""""""""""""""""""""
Stores all metrics in one file
"""""""""""""""""""""""""""""""""""""""


with open('output/results.txt', 'w') as f:
    for q, metric in metrics.items():
        print('q={}:'.format(q), file=f)
        for key, value in metric.items():
            print(key,'\n', value, file=f)

        
