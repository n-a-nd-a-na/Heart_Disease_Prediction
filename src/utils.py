import mlflow

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def log_model_metrics(model_name,params,accuracy,precision,recall,f1):
    mlflow.log_param('model',model_name)

    mlflow.log_params(params)

    mlflow.log_metric("accuracy", accuracy)      
    mlflow.log_metric("precision", precision)      
    mlflow.log_metric("recall", recall)      
    mlflow.log_metric("f1_score", f1)      

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)  
    precision = precision_score(y_test, y_pred)  
    recall = recall_score(y_test, y_pred)  
    f1 = f1_score(y_test, y_pred)  

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
    