if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd

@transformer
def transform(data, *args, **kwargs):
    """
    Transformer les données du fichier .parquet

    Args:
        data: Les données du .parquet (voir le bloc Dara Loader)
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Les données transformées
    """

    data.tpep_dropoff_datetime = pd.to_datetime(data.tpep_dropoff_datetime)
    data.tpep_pickup_datetime = pd.to_datetime(data.tpep_pickup_datetime)

    data['duration'] = data.tpep_dropoff_datetime - data.tpep_pickup_datetime
    data.duration = data.duration.dt.total_seconds() / 60

    data = data[(data.duration >= 1) & (data.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    data[categorical] = data[categorical].astype(str)

    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'