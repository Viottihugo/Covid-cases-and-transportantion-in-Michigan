# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:56:00 2020

@author: viott
"""
import pandas as pd
import numpy as np


def blight_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import GridSearchCV
    from datetime import datetime
    
    df = pd.read_csv('train.csv',  encoding = 'L1')
    df.set_index ('ticket_id', inplace=True)
    # Your code here
    #First, we'll discard all NaN in the Compliance column, and drop the columns that have almost no info other than NaN  
    df = df[df['compliance'].notnull()]    
    df=df[[ 'ticket_issued_date', 'hearing_date', 'fine_amount', 'admin_fee',
            'state_fee', 'late_fee', 'discount_amount', 'compliance']]
        
    #now we clear the NaN values
    #df.dropna(inplace=True)
    
    #df['violation_street_name'] = df['violation_street_name'].str.lower()
    #df['city'] = df['city'].str.lower()
    #df['state'] = df['state'].str.lower()
    #df['violation_code'] = df['violation_code'].str.lower()
    df['ticket_issued_date'] = df['ticket_issued_date'].values.astype(np.int64)
    df['hearing_date'] =  df['hearing_date'].values.astype(np.int64)
    X=df.iloc[:,:-1]
    
    X=X.dropna()
    
   
    y=df['compliance']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    
    #Rforest_clf = RandomForestClassifier(random_state = 0).fit(X_train, y_train)
    
    #y_score = Rforest_clf.predict(X_test)
        
    #fpr, tpr, _ = roc_curve(y_test, y_score)
    
    #roc_auc = auc(fpr, tpr)
    
    #print('roc_auc:  ',roc_auc)
    
    # roc_auc default:   0.642229285696
    #Rforest_clf = RandomForestClassifier(random_state = 0)
    #grid_values={'n_estimators': [3,5, 10, 20, 30, 50], 'max_depth': [3, 5, 7, 10]}
    
    #grid_clf_auc = GridSearchCV(Rforest_clf, param_grid = grid_values , scoring = 'roc_auc')
    #grid_clf_auc.fit(X_train, y_train)
    #y_decision_scores_auc = grid_clf_auc.predict(X_test)    
    #print('Test set AUC: ', roc_auc_score(y_test, y_decision_scores_auc))
    #print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
    #print('Grid best score (AUC): ', grid_clf_auc.best_score_)
    
    # Grid best parameter (max. AUC):  {'max_depth': 10, 'n_estimators': 50}
    # Grid best score (AUC):  0.802285761817
    
    clf = SVC(kernel='rbf', random_state=0)
    grid_values = {'gamma': [0.001, 0.01, 0.05, 0.1, 1, 10, 100]}

    # default metric to optimize over grid parameters: accuracy
    grid_clf_acc = GridSearchCV(clf, param_grid = grid_values, scoring = 'roc_auc')
    grid_clf_acc.fit(X_train, y_train)
    y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test)
    
    print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))
    print('Grid best parameter (max. AUC): ', grid_clf_auc.best_params_)
    print('Grid best score (AUC): ', grid_clf_auc.best_score_)
    
    
    
    return #_dummies# Your answer here

blight_model()