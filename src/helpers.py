import os
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from config import *

# Train Models
def train_model(pipe: Pipeline, params: dict, X_train, Y_train):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipe, params, cv=skf, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, Y_train)
    best_model = grid.best_estimator_
    return best_model

# Save Model
def save_model(model, file_name:str):
    try:
        os.makedirs('./models', exist_ok=True)
        file_path =f'./models/{file_name}.pkl'
        joblib.dump(model, file_path)
        
        if os.path.exists(file_path):
            print(f"Model saved successfully: {file_path}")
        else:
            print(f"Model not saved: {file_path}")
    except Exception as e:
        print(f"Error saving model {file_name}: {str(e)}")
        
        
# Evaluating Models
def evaluate(model_name:str, model, X_test, Y_test) -> dict:
    y_pred = model.predict(X_test)
    metrics ={
        'model_name': model_name,
        'accuracy': accuracy_score(Y_test, y_pred),
        'f1_score': f1_score(Y_test, y_pred, average='weighted'),
        'precision': precision_score(Y_test, y_pred, average='weighted'),
        'recall': recall_score(Y_test, y_pred, average='weighted')
    }
    return metrics


def load_data():
    df = pd.read_csv('./data/creditcard.csv')
    X = df.drop(['Class'],axis=1)
    Y = df['Class']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=TEST_SIZE,stratify=Y,random_state=RANDOM_STATE)
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_smote, Y_smote = smote.fit_resample(X_train, Y_train)
    
    return X_smote,Y_smote,X_test,Y_test