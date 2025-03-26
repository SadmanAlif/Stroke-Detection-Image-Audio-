import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
import time
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from pycaret.classification import *

hyperparameters_RFC = {'n_estimators': 300, 'max_depth': 90, 'min_samples_split': 6, 'min_samples_leaf': 3,
                       'max_features': 'sqrt', 'bootstrap': False, 'criterion': 'entropy'}

hyperparameters_XGB = {'max_depth': 9,
                       'min_child_weight': 1,
                       'learning_rate': 0.2,
                       'subsample': 0.8,
                       'colsample_bytree': 1.0,
                       'gamma': 0,
                       'n_estimators': 600,
                       'use_label_encoder': False,
                       'eval_metric': 'rmse',
                       'objective': 'binary:logistic'}

hyperparameters_CB = {'bagging_temperature': 0.8607305832563434, 'bootstrap_type': 'MVS',
                      'colsample_bylevel': 0.917411003148779,
                      'depth': 8, 'grow_policy': 'SymmetricTree', 'iterations': 918, 'l2_leaf_reg': 8,
                      'learning_rate': 0.29287291117375575, 'max_bin': 231, 'min_data_in_leaf': 9, 'od_type': 'Iter',
                      'od_wait': 21, 'one_hot_max_size': 7, 'random_strength': 0.6963042728397884,
                      'scale_pos_weight': 1.924541179848884, 'subsample': 0.6480869299533999}

cb = CatBoostClassifier(**hyperparameters_CB)
xgb = XGBClassifier(**hyperparameters_XGB, )
rf = RandomForestClassifier(**hyperparameters_RFC, random_state=150)

models = [
    ("XGB", xgb, "Blues"),
    ("RF", rf, "Purples"),
    ("CB", cb, "Reds"),
    ("MVC", VotingClassifier(estimators=[
        ("cb", cb),
        ('xgb', xgb),
        ('rf', rf),
    ], voting='hard'), "Greens")
]

def train_classifiers(data):
    print(data.shape)
    drop_columns = ['Filename', "is_stroke_face"]
    X = data.drop(drop_columns, axis=1)
    y = data["is_stroke_face"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=150)

    report_list = []

    for name, model, color in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        report_list.append([name,report])
    
    return report_list

'''
Training set = real(80%) + synthetic(100%)
Test set = real(20%)
'''
def train_classifier_combined(real, fake):

    real = real.drop('Filename', axis = 1)
    real_20 = real.sample(frac=0.2, random_state = 150)
    real_80 = real.drop(real_20.index)
    combined_df = pd.concat([real_80, fake], ignore_index=True)

    train_df = combined_df
    test_df = real_20

    print(train_df.head())
    print(test_df.head())


    X_train = train_df.iloc[:, :-1]  # All columns except the last one
    y_train = train_df.iloc[:, -1]   # Last column as target

    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]


    report_list = []

    for name, model, color in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        report_list.append([name,report])
    
    return report_list



"""
Trains models on 80% of real data and tests on 20% of synthetic data and vice versa. 
"""
def train_test_cross(train_data, test_data):

    if 'Filename' in train_data.columns:
        train_data = train_data.drop('Filename', axis=1)
    
    if 'Filename' in test_data.columns:
        test_data = test_data.drop('Filename', axis=1)
    
    train_data = train_data.sample(frac = 0.8, random_state = 150)
    test_data = test_data.sample(frac = 0.2, random_state = 150)

    print(train_data.head())
    print(test_data.head())

    X_train = train_data.iloc[:, :-1]  # All columns except the last one
    y_train = train_data.iloc[:, -1]   # Last column as target

    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]


    for name, model, color in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        report_list.append([name,report])
    
    return report_list


real_df =  pd.read_csv('COMBO2.csv')
sythetic_df =  pd.read_csv('synthetic.csv')
combined_df = pd.concat([real_df, sythetic_df], ignore_index=True)


report_list = train_classifier_combined(real_df, sythetic_df)

for name,report in report_list:
    print(f'Classification report for {name}')
    print(report)




