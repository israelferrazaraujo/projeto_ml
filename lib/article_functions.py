import numpy as np

def D_l(e_i, G_k, d):
  """
  Measures the local matching between an example e_i \in C_k and the cluster
  prototype G_k \in E^{(q)}.

  Page 3, expression (2).
  """
  return np.sum(np.take(np.take(d, e_i, axis=0), G_k, axis=1), axis=1)

def D_g(e_i, G_k, L_k, D, s=1):
  """
  Matching function.
  Measures the global matching between an example e_i \in C_k and the cluster
  prototype G_k \in E^{(q)}, parameterized by both the parameter s and the 
  vector of relevance weights L_k.

  Page 7, expression (7).
  """
  p = len(L_k) # =len(D) -> len of dissimilarity matrices.
  return np.dot(np.power(L_k,s), [ D_l(e_i, G_k, D[j]) for j in range(p) ])
  
def adequacy(G, D, U, L, m=1.6, s=1):
  """
  The adequacy criterion.

  Page 6, expression (6).
  """
  n = len(U)
  K = len(G)
  return np.sum(
            np.multiply( 
                np.power(U,m), np.transpose(
                    [D_g(range(n), G[k], L[k], D, s=s) for k in range(K)]) 
                )
            )

def prototype(k, U, D, L, q=4, m=1.6, s=1): # k=1, ..., K 
  """
  Step 1: Computation of the best prototypes.
  In this step, the fuzzy partition represented by U = (u_1, ... , u_n) and 
  the vector of relevance weight vectors L=(l_1, ..., l_K) are fixed.

  Page 6, proposition 2.3.
  """
  G_k = np.empty(0, dtype=np.int)
  U_k = np.power(U[:,k], m)
  L_k = np.power(L[k,:], s)
  #D_T = np.transpose(D, axes=(1,0,2))
  #j = np.dot( U_k, np.dot(L_k, D_T )) 
  n = len(U_k)
  opt_h = np.array(range(n))
  j = np.array([ np.dot( U_k, 
                  np.dot(L_k, D[:][:,h])) 
                for h in opt_h ])
  while (len(G_k) < q):
    index = np.argmin(j[ opt_h ])
    l = opt_h[index]
    opt_h = np.delete(opt_h, index)
    G_k = np.append(G_k, l)
    
  return G_k

def weights(U_k, G_k, D, m=1.6):
  """
  Step 2:  Computation of the best relevance weight vector.
  In this step, the fuzzy partition represented by U = (u1, ... , un) and the
  vector of prototypes G = (G1, ... , G K ) are fixed.
  
  Proposition 2.4. The vectors of weights are computed according to the
  matching function used.

  Page 7, expression (9).
  """
  n = len(U_k)
  p = len(D)
  
  p_U_k = np.power(U_k,m)
  D_l_p = [np.transpose([D_l(range(n), G_k, D[h])]) for h in range(p)]
  dot_p = [np.dot(p_U_k, D_l_p[h])[0] for h in range(p)]

  num = np.prod([dot_p[h] for h in range(p)])**(1/p)

  L_k = [num / dot_p[j] for j in range(p)]

  return np.array(L_k)

def fuzzy_partition(G, D, L, m=1.6, s=1):
  """
  Step 3: Definition of the best fuzzy partition.
  In this step, the vector of prototypes G = (G1, ... , G K ) and the vector 
  of relevance weight vectors L = (l1, ... , lK ) are fixed.

  Page 8, proposition 2.5, expression (11).
  """
  n = len(D[0][0]) # dim. da matriz de dissimilaridade é (n x n).
  K = len(G)

  D_g_K = [ D_g(range(n), G[h], L[h], D, s=s) for h in range(K) ]

  with np.errstate(divide='ignore', invalid='ignore'):
    U = np.transpose(
        [
          np.power(
              np.sum(
                  np.power(
                    np.divide(D_g_K[k] , D_g_K)
                  , (1/(m-1))) 
              , axis=0) , -1)
          for k in range(K)]
        )
  
  # Tratamento das indefinições e indeterminações.
  # Podem ser eliminadas pela condição de que a soma em K é sempre igual a 1 .
  # Os problemas ocorrem quando e_i = e, então teremos apenas um item igual
  # a 1.0 e o restante iguais a 0.0 .
  U[np.isnan(U)] = 1.0 # U[np.where(np.isnan(U) == True)] = 1.0

  return U

