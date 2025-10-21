import pandas as pd
from helpers import train_model, save_model, evaluate, load_data
from config import *
from pipeline import pipeline

def train():
    X_train, Y_train, X_test, Y_test = load_data()
    
    results = []
    
    for model_name,model_def in models.items():
        best_model = None
        best_score = 0
        params = model_def['params']
        pipe = pipeline(model_name)
        print(f"Starting Model: {model_name}")
        trained_model = train_model(pipe, params, X_train, Y_train)
        metrics = evaluate(model_name,trained_model,X_test,Y_test)
        results.append(metrics)
        cv_score = metrics['f1_score']
        if cv_score > best_score:
            best_score = cv_score
            best_model = trained_model
            best_model_filename = f"{model_name}"
        if best_model:
            save_model(best_model, best_model_filename)
            print(f"Model saved: {best_model_filename}.pkl")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1_score", ascending=False)
    results_df.to_csv('./reports/models_report.csv', index=True)
    print("Evaluation Results saved to /reports/model_evaluation.csv")
    
    
if __name__=="__main__":
    train()