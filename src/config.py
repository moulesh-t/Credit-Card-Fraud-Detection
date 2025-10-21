from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


TEST_SIZE = 0.2

RANDOM_STATE = 42

RF_PARAMS = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, None],
    'model__criterion': ['gini', 'entropy'],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2]
}

XGB_PARAMS = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.05, 0.1, 0.2],
    'model__subsample': [0.7, 0.8],
    'model__colsample_bytree': [0.7, 0.8]
    }

LGBM_PARAMS = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10, -1],
    'model__learning_rate': [0.05, 0.1],
    'model__num_leaves': [31, 50],
    'model__subsample': [0.8, 0.9]
}

models = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE),
        'params': RF_PARAMS
    },
    'Gradient Boosting': {
                          'model': XGBClassifier(),
                          'params': XGB_PARAMS
    },
    'Light Boosting': {
        'model': LGBMClassifier(),
        'params': LGBM_PARAMS
    }
}
    