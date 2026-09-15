from preprocess import load_data,preprocess_data
from utils import evaluate_model,log_model_metrics

from sklearn.preprocessing import StandardScaler
import joblib

df = load_data('data/heart_processed.csv')

X,y = preprocess_data(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, 'model/scaler.pkl')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42, stratify = y)

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import mlflow
import mlflow.sklearn

mlflow.set_experiment('Heart_Disease_Prediction')

mlp_params = [{'hidden_layer_sizes': (50,),'max_iter': 500, 'learning_rate_init': 0.001},{'hidden_layer_sizes': (100,), 'max_iter': 500, 'learning_rate_init': 0.001},{'hidden_layer_sizes': (100, 50), 'max_iter': 500, 'learning_rate_init': 0.001},{'hidden_layer_sizes': (100, 50), 'max_iter': 700, 'learning_rate_init': 0.0001}]

for i, params in enumerate(mlp_params):
    mlp = MLPClassifier(**params, random_state=42)

    with mlflow.start_run(run_name=f'MLP {i+1}'):
        mlp.fit(X_train,y_train)
        y_pred = mlp.predict(X_test)
        metrics = evaluate_model(y_test,y_pred)

        print(f'/nMLP {i+1} Results')
        print(params)
        print(metrics)

        log_model_metrics('MLP',params,metrics['accuracy'],metrics['precision'],metrics['recall'],metrics['f1_score'])

        mlflow.sklearn.log_model(mlp,name='model',skops_trusted_types=['sklearn.neural_network._stochastic_optimizers.AdamOptimizer'])

svm_params = [{'kernel': 'rbf', 'C': 1, 'gamma': 'scale','probability': True},{'kernel': 'rbf', 'C': 10, 'gamma': 'scale','probability': True},{'kernel': 'rbf', 'C': 100, 'gamma': 'scale','probability': True},{'kernel': 'linear', 'C': 1,'probability': True},{'kernel': 'linear', 'C': 10,'probability': True}]

for i, params in enumerate(svm_params):
    svm = SVC(**params)

    with mlflow.start_run(run_name = f'SVM {i+1}'):
        svm.fit(X_train,y_train)
        y_pred = svm.predict(X_test)
        metrics = evaluate_model(y_test,y_pred)

        print(f'\nSVM {i+1} Results')
        print(params)
        print(metrics)

        log_model_metrics('SVM',params,metrics['accuracy'],metrics['precision'],metrics['recall'],metrics['f1_score'])

        mlflow.sklearn.log_model(svm,name ='model')


rf_params = [{'n_estimators': 100, 'max_depth': None},{'n_estimators': 200, 'max_depth': None},{'n_estimators': 200, 'max_depth': 10},{'n_estimators': 300, 'max_depth': 15}]

for i, params in enumerate(rf_params):
    rf = RandomForestClassifier(**params,random_state=42)

    with mlflow.start_run(run_name=f'Random Forest {i+1}'):
        rf.fit(X_train,y_train)
        y_pred = rf.predict(X_test)
        metrics = evaluate_model(y_test,y_pred)

        print(f'\nRandom Forest {i+1} Results')
        print(params)
        print(metrics)

        log_model_metrics('Random Forest',params,metrics['accuracy'],metrics['precision'],metrics['recall'],metrics['f1_score'])

        mlflow.sklearn.log_model(rf,name = 'model')


nb_params = [{'var_smoothing': 1e-9},{'var_smoothing': 1e-8},{'var_smoothing': 1e-7},{'var_smoothing': 1e-6},{'var_smoothing': 1e-5}]

for i, params in enumerate(nb_params):
    nb = GaussianNB(**params)

    with mlflow.start_run(run_name = f'Naive Bayes {i+1}'):
        nb.fit(X_train,y_train)
        y_pred = nb.predict(X_test)
        metrics = evaluate_model(y_test,y_pred)

        print(f'\nNaive Bayes {i+1} Results')
        print(params)
        print(metrics)

        log_model_metrics('Naive Bayes',params,metrics['accuracy'],metrics['precision'],metrics['recall'],metrics['f1_score'])

        mlflow.sklearn.log_model(nb,name = 'model')

