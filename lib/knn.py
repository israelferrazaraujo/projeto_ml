import numpy as np


"""""""""""""""""""""""""""""""""""""""
select view 
"""""""""""""""""""""""""""""""""""""""

def select_view(data, view):
    ''' function to select view data
    '''
    cols = [x for x in data.columns if view in x ]
    return data[cols]

"""""""""""""""""""""""""""""""""""""""
KNN mixture view model
"""""""""""""""""""""""""""""""""""""""

def KnnViewModelling(train_data, test_data, K):
    ''' build and run models for each view,
        combine probabilities of models for the final
        decision, return accuracy of model  
    '''
    
    
    # split features and target
    features_train = train_data.drop(['target', 'kfold'], axis=1)
    target_train = train_data['target']
    features_test = test_data.drop(['target', 'kfold'], axis=1)
    target_test = test_data['target']

    # build models for each view and return probabilities
    from sklearn.neighbors import KNeighborsClassifier
    knn_view1 = KNeighborsClassifier(n_neighbors=K, weights='distance')
    knn_view1.fit(select_view(features_train, 'fac'), target_train)
    pred_knn_view1 = list(knn_view1.predict_proba(select_view(features_test,  'fac')))

    knn_view2 = KNeighborsClassifier(n_neighbors=K, weights='distance')
    knn_view2.fit(select_view(features_train, 'fou'), target_train)
    pred_knn_view2 = list(knn_view2.predict_proba(select_view(features_test,  'fou')))

    knn_view3 = KNeighborsClassifier(n_neighbors=K, weights='distance')
    knn_view3.fit(select_view(features_train, 'kar'), target_train)
    pred_knn_view3 = list(knn_view3.predict_proba(select_view(features_test,  'kar')))

    # probability combination
    predictions = []
    for i in range(len(features_test)):
        # sum probs
        sum_prob = pred_knn_view1[i]+pred_knn_view2[i]+pred_knn_view3[i]
        # normalize
        norm_prob = (sum_prob - sum_prob.min()) / (sum_prob - sum_prob.min()).sum()
        # decision
        decision = np.where(norm_prob == max(norm_prob))[0][0]
        predictions.append(decision)

    # evaluate decision
    correct_shots = [1 if predictions[i] ==  target_test[i] else 0 for i in range(len(target_test))]
    accuracy = sum(correct_shots)/len(correct_shots)
    
    return accuracy

