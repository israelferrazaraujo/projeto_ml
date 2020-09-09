import numpy as np

def MPC(U):
  """
  Modified Partition Coefficient (MPC)

  The range of Modified Partition Coefficient (MPC) is the unit interval [0, 1],
  where MPC=0 corresponds to maximum fuzziness and MPC=1 to a hard partition.

  ref. "Pattern Recognition Letters", Suleman (2015).
  """
  n, K = U.shape
  return (K/(K-1)) * (np.sum(np.power(U, 2)) / n) - (1/(K-1))

def PE(U):
  """
  Partition Entropy (PE)

  The PE index is a scalar measure of the amount of fuzziness in a given U.
  The PE index values range in [0, log(C)], the closer the value of PE to 0,
  the crisper the clustering is. The index value close to the upper bound 
  indicates the absence of any clustering structure in the datasets or inability
  of the algorithm to extract it.
  """
  with np.errstate(divide='ignore'): # quando u_ik=0 gera erro de divisão.
    log_U = np.log(U)                # quando u_ik=0, log(u_ik)=inf.
  log_U[np.isinf(log_U)] = 1 # qualquer valor, eles serão multiplicados por zero.
  
  n,_ = U.shape
  return -(1/n)*np.sum( np.multiply(U, log_U) )

def traditional_f_measure(Pi, Pk):
  q = precision(Pi, Pk)+recall(Pi, Pk)
  if (q == 0):
    return 0

  return 2*( (precision(Pi, Pk)*recall(Pi, Pk)) / q )
  
def f_measure(P, Q):
  import numpy as np

  N = len(P)

  unique, counts = np.unique(P, return_counts=True)
  b = dict(zip(unique, counts))

  return (1/N) * np.sum( [ b[i]*np.max( [ traditional_f_measure(P==i, Q==k) for k in b.keys()] ) for i in b.keys()] ) 

def OERC(P, Q):
  """
  Overall Error Rate of Classification (OERC)
  """
  N = len(P)
  unique, counts = np.unique(P, return_counts=True)
  b = dict(zip(unique, counts))
  
  return 1-(1/N)* np.sum( [ np.max([ true_positives(P==i, Q==k) for i in b.keys()]) for k in b.keys() ])




"""""""""""""""""""""""""""""""""""""""
Auxiliary functions to calculate OERC
"""""""""""""""""""""""""""""""""""""""




def true_positives(Pi, Pk):
  return np.sum([x and y for x,y in zip(Pi, Pk)])

def false_positives(Pi, Pk):
  return np.sum([not x and y for x,y in zip(Pi, Pk)])
  
def false_negatives(Pi, Pk):
  return np.sum([x and not y for x,y in zip(Pi, Pk)])
  
def precision(Pi, Pk):
  q = true_positives(Pi, Pk)+false_positives(Pi, Pk)
  if (q == 0):
    return 0

  return true_positives(Pi, Pk) / q

def recall(Pi, Pk):
  q = true_positives(Pi, Pk)+false_negatives(Pi, Pk)
  if (q == 0):
    return 0

  return true_positives(Pi, Pk) / q




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Auxiliary functions to calculate and format the metrics
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""



def calc_metric(U, G, y):
    from sklearn.metrics import adjusted_rand_score

    K = len(G)
    crisp = np.argmax(U, axis=1)

    result = {'crisp_partition_objects': {i: list(np.where(crisp == i)) for i in range(K)},
              'prototypes' : {i:list(g) for i, g in enumerate(G)} ,
              'objetcs_per_crisp_partition' : {i: np.count_nonzero(crisp == i, axis=0) for i in range(K)} ,
              'modified_partition_coefficient' : MPC(U) ,             # between 0 (worse)  and 1      (better)
              'partition_entropy' : PE(U) ,                           # between 0 (better) and log(C) (worse)
              'adjusted_rand_score' : adjusted_rand_score(y, crisp) , # between 0 (worse)  and 1      (better)
              'f-measure': f_measure(y, crisp),                       # between 0 (worse)  and 1      (better)
              'OERC' : OERC(y, crisp) ,                               # between 0 (better) and 1      (worse)
              'crisp_partition' : list(crisp)}

    return result




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Additional experiments
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""




def attrib_translation(y, crisp):
  from itertools import permutations
  from sklearn.metrics import accuracy_score

  crisp_labels = list(set(crisp))
  crisp_labels.sort()

  indexes = [(j, [i for i, x in enumerate(crisp) if x == j]) for j in crisp_labels]
  translation_labels = [ max(list(y[index]), key=list(y[index]).count) for j, index in indexes ]
  translation_table = dict((x, y) for x, y in zip(crisp_labels, translation_labels)) 

  translated_crisp = np.empty(len(crisp), dtype=int)
  for j, index in indexes:
    translated_crisp[index] = translation_table[j]
  
  error = 1-accuracy_score(y, translated_crisp)
    
  return error, translation_table, list(translated_crisp)




