if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

@data_exporter
def export_data(data, *args, **kwargs):

    # Variables catégorielles
    categorical = ['PULocationID', 'DOLocationID']

    # Conversion en dictionnaires
    train_dicts = data[categorical].to_dict(orient='records')

    # Vectorisation
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    # Variable cible
    y_train = data['duration'].values

    # Entraînement du modèle
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Affichage de l'intercept
    print(f"Intercept du modèle : {lr.intercept_}")