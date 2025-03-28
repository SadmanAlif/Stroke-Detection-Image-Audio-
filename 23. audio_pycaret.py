from pycaret.classification import *  
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_pycaret(data, target_column):
    clf_setup = setup(data, target=target_column, session_id=123)
    best_model = compare_models()
    tuned_model = tune_model(best_model)
    predictions = predict_model(tuned_model)
    return predictions

def train_rf(data, target_column='is_stroke_face'):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150)
    rf_model = RandomForestClassifier(random_state=123)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, rf_predictions)
    print("Random Forest Accuracy:", accuracy)
    return rf_model, rf_predictions

def train_xgb(data, target_column='is_stroke_face'):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150)
    xgb_model = XGBClassifier(random_state=123)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, xgb_predictions)
    print("XGBoost Accuracy:", accuracy)
    return xgb_model, xgb_predictions


data = pd.read_csv("MFCCoutput.csv")
data = data.drop(data.columns[0], axis = 1)
train_pycaret(data, 'class')
