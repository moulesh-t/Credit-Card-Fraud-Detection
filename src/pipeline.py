from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from config import *

# Create Pipelines
def pipeline(model_name):
    if model_name not in models.keys():
        raise ValueError('Invalid Model name provided')
    else:
        model = models[model_name]['model']
        pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('model', model)
        ])
        return pipe
