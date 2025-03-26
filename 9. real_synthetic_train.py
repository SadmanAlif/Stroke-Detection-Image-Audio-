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


drop_columns = ['rightEyeLower1_130', 'rightEyeLower1_25', 'rightEyeLower1_110',
                'rightEyeLower1_24', 'rightEyeLower1_23', 'rightEyeLower1_22', 'rightEyeLower1_26',
                'rightEyeLower1_112', 'rightEyeLower1_243', 'rightEyeUpper0_246', 'rightEyeUpper0_161',
                'rightEyeUpper0_160', 'rightEyeUpper0_159', 'rightEyeUpper0_158', 'rightEyeUpper0_157',
                'rightEyeUpper0_173', 'rightEyeLower0_33', 'rightEyeLower0_7', 'rightEyeLower0_163',
                'rightEyeLower0_144', 'rightEyeLower0_145', 'rightEyeLower0_153', 'rightEyeLower0_154',
                'rightEyeLower0_155', 'rightEyeLower0_133', 'leftEyeLower3_372', 'leftEyeLower3_340',
                'leftEyeLower3_346', 'leftEyeLower3_347', 'leftEyeLower3_348', 'leftEyeLower3_349', 'leftEyeLower3_350',
                'leftEyeLower3_357', 'leftEyeLower3_465', 'rightEyeLower2_226', 'rightEyeLower2_31',
                'rightEyeLower2_228', 'rightEyeLower2_229', 'rightEyeLower2_230', 'rightEyeLower2_231',
                'rightEyeLower2_232', 'rightEyeLower2_233', 'rightEyeLower2_244', 'rightEyeUpper2_113',
                'rightEyeUpper2_225', 'rightEyeUpper2_224', 'rightEyeUpper2_223', 'rightEyeUpper2_222',
                'rightEyeUpper2_221', 'rightEyeUpper2_189', 'leftEyeUpper1_467', 'leftEyeUpper1_260',
                'leftEyeUpper1_259', 'leftEyeUpper1_257', 'leftEyeUpper1_258', 'leftEyeUpper1_286', 'leftEyeUpper1_414',
                'leftEyeLower2_446', 'leftEyeLower2_261', 'leftEyeLower2_448', 'leftEyeLower2_449', 'leftEyeLower2_450',
                'leftEyeLower2_451', 'leftEyeLower2_452', 'leftEyeLower2_453', 'leftEyeLower2_464', 'leftEyeLower1_359',
                'leftEyeLower1_255', 'leftEyeLower1_339', 'leftEyeLower1_254', 'leftEyeLower1_253', 'leftEyeLower1_252',
                'leftEyeLower1_256', 'leftEyeLower1_341', 'leftEyeLower1_463', 'leftEyeUpper2_342', 'leftEyeUpper2_445',
                'leftEyeUpper2_444', 'leftEyeUpper2_443', 'leftEyeUpper2_442', 'leftEyeUpper2_441', 'leftEyeUpper2_413',
                'rightEyebrowLower_35', 'rightEyebrowLower_124', 'rightEyebrowLower_46', 'rightEyebrowLower_53',
                'rightEyebrowLower_52', 'rightEyebrowLower_65', 'leftEyebrowLower_265', 'leftEyebrowLower_353',
                'leftEyebrowLower_276', 'leftEyebrowLower_283', 'leftEyebrowLower_282', 'leftEyebrowLower_295',
                'rightEyeUpper1_247', 'rightEyeUpper1_30', 'rightEyeUpper1_29', 'rightEyeUpper1_27',
                'rightEyeUpper1_28', 'rightEyeUpper1_56', 'rightEyeUpper1_190', 'leftEyeUpper0_466',
                'leftEyeUpper0_388', 'leftEyeUpper0_387', 'leftEyeUpper0_386', 'leftEyeUpper0_385', 'leftEyeUpper0_384',
                'leftEyeUpper0_398', 'noseBottom_2', 'midwayBetweenEyes_168', 'noseRightCorner_98',
                'noseLeftCorner_327']

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
All regions
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



'''
Top regions
Training set = real(80%) + synthetic(100%)
Test set = real(20%)
'''
def train_classifier_combined_top_regions(real, fake):

    real = real.drop(columns = 'Filename', axis = 1)
    real = real.drop(drop_columns, axis = 1)

    real_20 = real.sample(frac=0.2, random_state = 150)
    real_80 = real.drop(real_20.index)

    fake = fake.drop(drop_columns, axis = 1)
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

    report_list = []

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
synthetic_df =  pd.read_csv('synthetic.csv')

report_list = train_classifier_combined_top_regions(real_df, synthetic_df)

for name,report in report_list:
    print(f'Classification report for {name}')
    print(report)




