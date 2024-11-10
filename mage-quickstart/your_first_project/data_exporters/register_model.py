if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import mlflow
import mlflow.sklearn
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

@data_exporter
def export_data(*args, **kwargs):
    """
    Enregistre le modèle et le vectoriseur dans MLFlow avec les métriques de performance.
    """

    # Charger les données transformées depuis le fichier temporaire
    # Etape ajustée car sinon erreur "dataframe"
    data = pd.read_parquet('/tmp/transformed_data.parquet')

    # Configuration MLFlow
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Chargement du modèle et du vectoriseur précédemment sauvegardés
    lr = load('linear_model.pkl')
    dv = load('dict_vectorizer.pkl')

    # Préparation des données pour calculer les métriques
    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = data[categorical].to_dict(orient='records')
    X_train = dv.transform(train_dicts)
    y_train = data['duration'].values

    # Calcul des prédictions pour les métriques
    y_pred = lr.predict(X_train)

    # Calcul des métriques
    mae = mean_absolute_error(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)

    # Enregistrement dans MLFlow
    with mlflow.start_run():
        # Enregistrer le modèle en tant que modèle MLFlow
        mlflow.sklearn.log_model(lr, "model")
        
        # Enregistrer le vectoriseur en tant qu'artefact dans MLFlow
        mlflow.log_artifact('dict_vectorizer.pkl', "dict_vectorizer")
        
        # Enregistrer l’intercept comme paramètre
        mlflow.log_param("intercept", lr.intercept_)

        # Enregistrer les métriques
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)

        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
        print("Modèle, vectoriseur et métriques enregistrés dans MLFlow.")