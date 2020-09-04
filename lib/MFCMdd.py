from .article_functions import *

def MFCMdd_RWL_P(D, K=10, m=1.6, T=150, e=10**(-10), s=1, q=2, reps=100):
    """
    Partitioning fuzzy K-medoids clustering algorithms with relevance weight for
    each dissimilarity matrix estimated locally (described on page 9).

    variáveis especificadas no projeto:
    K = 10          # number of clusters.
    m = 1.6         # adjust factor.
    T = 150         # iteration limit.
    e = 10**(-10)   # stopping criterion.
    s = 1           # defined as 1.0 at item 2.2.2.(a).
    q = 2           # cardinality.
    """

    p = len(D)      # views number.
    n = len(D[0])   # number of dataset objects.

    J_min = 2**32
    result = None
    for i in range(reps):
        print('\nRepetition {}'.format(i))

        """""""""""""""""""""
        Article's algorithm
        """"""""""""""""""""" 
        t, L, G, U, J = initialization(D, K, m, s, q, p, n)
        while (True):
            t = t + 1
            G = step1(D, U, L, K, m, s, q)
            L = step2(D, U, G, K, m)
            U = step3(D, G, L, m, s)
            J, diff = step4(D, U, G, L, J, m, s, t)
            if (diff <= e or t >= T):
                break
        """"""""""""""""""

        # stores the best result
        if (J < J_min):
            J_min = J
            result = (U, G)

    return result


"""""""""""""""""""""""""""""""""
Steps defined in the article
"""""""""""""""""""""""""""""""""


import numpy as np
import random

def initialization(D, K, m, s, q, p, n):
    """
    Variables initialization
    """
    # orientações do artigo
    t = 0
    L = np.ones([K, p])
    G =  [ np.array(random.sample(range(n), q)) for i in range(K) ] # randomly select K distinct prototypes. Random sampling without replacement. Não supervisionado.
    #G =  [ np.array(random.sample(range(n//K), q))+(n//K)*i for i in range(K) ] # randomly select K distinct prototypes. Semi supervisionado.
    U = fuzzy_partition(G, D, L, m=m, s=s)
    J = adequacy(G, D, U, L, m=m, s=s)

    return t, L, G, U, J

def step1(D, U, L, K, m, s, q):
    """
    Computation of the best prototypes
    """
    return [prototype(k, U, D, L, q=q, m=m, s=s) for k in range(K)]

def step2(D, U, G, K, m):
    """
    Computation of the best relevance weight vector
    """
    return np.array([weights(U[:,k], G[k], D, m=m) for k in range(K)])

def step3(D, G, L, m, s):
    """
    Definition of the best fuzzy partition
    """
    return fuzzy_partition(G, D, L, m=m, s=s)

def step4(D, U, G, L, J, m, s, t):
    """
    Stopping criterion
    """
    J_minus = J
    J = adequacy(G, D, U, L, m=m, s=s)
    diff = np.abs(J - J_minus)

    print('Iter {}: |{:.7e} - {:.7e}| = {:.7e}'.format(t, J, J_minus, diff))

    return J, diff


