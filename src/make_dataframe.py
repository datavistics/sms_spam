from pathlib import Path

import pandas as pd

project_dir = Path(__file__).parents[2]


def make_dataframe():
    """
    Create dataframe from tsv file
    :return:
    """
    data_path = project_dir / 'data' / 'raw' / 'SMSSpamCollection'
    df = pd.read_csv(data_path, sep='\t', names=['label', 'text'])
    return df


if __name__ == '__main__':
    make_dataframe()
