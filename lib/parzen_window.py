import pandas as pd
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None   # define assignment not chained for preprocessing  

"""""""""""""""""""""""""""""""""""""""
Parzen Window multiclass classification 
based on Kernel Density Estimation
Source: https://qastack.com.br/stats/244012/can-you-explain-parzen-window-kernel-density-estimation-in-laymans-terms
"""""""""""""""""""""""""""""""""""""""

class ParzenWindowMulticlassKDE():
    

    def __init__(self):
        self.models = {}


    def train(self, data, parameter):
        ''' calculate gaussian distribution in parzen window density 
            for each class of target and save models in self.class
        '''
        from sklearn.neighbors import KernelDensity
        for classe in data['target'].unique():
            data_class = data[data['target'] == classe]
            data_class.drop('target', axis=1, inplace=True)
            parzen_model = KernelDensity(bandwidth=parameter)
            parzen_model.fit(data_class.values)
            self.models[classe] = parzen_model
    

    def predict_proba(self, test_data):
        ''' get estimation probability for each parzen model and
            normalize probs between 0 and 1 
        '''
        final_preds =[]
        models_preds = {}

        # capture probs
        for item in self.models.keys(): 
            models_preds[item] = np.exp(self.models[item].score_samples(test_data))

        # normalization
        for i in range(len(models_preds[0])):
            each_pred = {}
            for preds in models_preds.keys():
                each_pred[preds] = models_preds[preds][i]
            final_preds.append(each_pred)
        return final_preds
    

    def predict(self, test_data):
        ''' return predicted decision based on 
            maximum probability class
        '''
        decision = []
        pred_array = self.predict_proba(test_data)

        for pred in pred_array:
            decision_index = np.where(pred == max(pred))[0][0]
            decision.append(decision_index)
        return decision



"""""""""""""""""""""""""""""""""""""""
Parzen Window mixture view model
"""""""""""""""""""""""""""""""""""""""


def select_view_parzen(data, view, train=True):
    ''' function to select view data 
        if train=True, target is added
    '''
    cols = [x for x in data.columns if view in x ]
    if train == True:
        cols.append('target')
    return data[cols]



def ParzenViewModelling(train_data, validation_data, parameter):
    ''' build and run models for each view,
        combine probabilities of models for the final
        decision, return accuracy of model  
    '''

    target_test = validation_data['target']

    # build models for each view and return probabilities
    parzen_view1 =  ParzenWindowMulticlassKDE()
    parzen_view1.train(select_view_parzen(train_data, 'fac'), parameter)
    pred_parzen_view1 = parzen_view1.predict_proba(select_view_parzen(validation_data,  'fac', train=False))

    parzen_view2 = ParzenWindowMulticlassKDE()
    parzen_view2.train(select_view_parzen(train_data, 'fou'), parameter)
    pred_parzen_view2 = list(parzen_view2.predict_proba(select_view_parzen(validation_data,  'fou', train=False)))

    parzen_view3 = ParzenWindowMulticlassKDE()
    parzen_view3.train(select_view_parzen(train_data, 'kar'), parameter)
    pred_parzen_view3 = list(parzen_view3.predict_proba(select_view_parzen(validation_data,  'kar', train=False)))

    # probability combination
    predictions = []
    for i in range(len(validation_data)):
        view_combination = {}
        # sum exponential probability
        for classe in pred_parzen_view1[i].keys():
            view_combination[classe] = pred_parzen_view1[i][classe] + pred_parzen_view2[i][classe] + pred_parzen_view3[i][classe] 
        keys = list(view_combination.keys())
        values = list(view_combination.values())
        # normalization
        values = (values - min(values)) / (values - min(values) ).sum()
        final_probabilities = {}
        cont=0
        for key in keys:
            final_probabilities[key] = values[cont]
            cont+=1

        # decision
        decision = max(final_probabilities, key=final_probabilities.get)
        predictions.append(decision)

    # evaluate decision
    correct_shots = [1 if predictions[i] ==  target_test[i] else 0 for i in range(len(target_test))]
    accuracy = sum(correct_shots)/len(correct_shots)
    
    return accuracy

