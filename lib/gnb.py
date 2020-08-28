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
Gaussian mixture view model
"""""""""""""""""""""""""""""""""""""""

def GnbViewModelling(train_data, test_data):
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
    from sklearn.naive_bayes import GaussianNB
    gnb_view1 = GaussianNB()
    gnb_view1.fit(select_view(features_train, 'fac'), target_train)
    pred_gnb_view1 = list(gnb_view1.predict_proba(select_view(features_test,  'fac')))

    gnb_view2 = GaussianNB()
    gnb_view2.fit(select_view(features_train, 'fou'), target_train)
    pred_gnb_view2 = list(gnb_view2.predict_proba(select_view(features_test,  'fou')))

    gnb_view3 = GaussianNB()
    gnb_view3.fit(select_view(features_train, 'kar'), target_train)
    pred_gnb_view3 = list(gnb_view3.predict_proba(select_view(features_test,  'kar')))

    # probability combination
    predictions = []
    for i in range(len(features_test)):
        # sum probs
        sum_prob = pred_gnb_view1[i]+pred_gnb_view2[i]+pred_gnb_view3[i]
        # normalize
        norm_prob = (sum_prob - sum_prob.min()) / (sum_prob - sum_prob.min()).sum()
        # decision
        decision = np.where(norm_prob == max(norm_prob))[0][0]
        predictions.append(decision)

    # evaluate decision
    correct_shots = [1 if predictions[i] ==  target_test[i] else 0 for i in range(len(target_test))]
    accuracy = sum(correct_shots)/len(correct_shots)
    
    return accuracy

