if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom

import mlflow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image, display
import os

@custom
def transform_custom(*args, **kwargs):
    # Configurer l'URI de suivi de MLFlow
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Récupérer toutes les exécutions des experiments 'Default' dans MLFlow
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    runs = client.search_runs(experiment.experiment_id)

    # Extraire les métriques de chacune d'elle
    metrics = []
    for run in runs:
        metrics.append({
            "run_id": run.info.run_id,
            "mae": run.data.metrics.get("mae", None),
            "mse": run.data.metrics.get("mse", None),
            "rmse": run.data.metrics.get("rmse", None),
        })

    # Convertir en DataFrame pour un affichage graphique
    df_metrics = pd.DataFrame(metrics)
    df_metrics = df_metrics.dropna()  # Supprimer les lignes avec des valeurs None si nécessaire

    # Tracer les métriques
    # Ici j'utilise seaborn car matplotlib ne me génère pas le graphique (conflit de version peut être)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_metrics, x="run_id", y="mae", marker='o', label="MAE")
    sns.lineplot(data=df_metrics, x="run_id", y="mse", marker='o', label="MSE")
    sns.lineplot(data=df_metrics, x="run_id", y="rmse", marker='o', label="RMSE")
    plt.xlabel("Run ID")
    plt.ylabel("Metric Value")
    plt.title("Performance Metrics for Model Runs")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Créer le dossier outputs s'il n'existe pas afin d'enregistrer les images des graphiques
    output_dir = "your_first_project/outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Enregistrement du graphique dans un fichier image dans le dossier persistant
    image_path = os.path.join(output_dir, "performance_metrics_seaborn.png")
    plt.savefig(image_path)
    plt.close()

    # Afficher l'image directement dans l'interface
    # Ne s'affiche pas donc voir le dossier outputs
    #display(Image(filename=image_path))

    return df_metrics 