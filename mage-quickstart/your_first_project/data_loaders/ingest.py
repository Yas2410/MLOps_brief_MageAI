if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd

@data_loader
def load_data(*args, **kwargs):
    """
    Bloc pour charger les données de Yellow Taxis - March 2023

    Returns:
        Un dataframe avec les données du .parquet
    """
    file_path = "your_first_project/Brief_03/data/yellow_tripdata_2023-03.parquet"
    data = pd.read_parquet(file_path)

    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
