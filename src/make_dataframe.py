import zipfile
from pathlib import Path

import pandas as pd
import requests

project_dir = Path(__file__).parents[2]
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
data_path = project_dir / 'data' / 'smsspamcollection.zip'
file_path = project_dir / 'data' / 'raw' / 'SMSSpamCollection'


def download_data():
    """
    Download project data
    :return:
    """
    r = requests.get(url, allow_redirects=True)
    open(data_path, 'wb').write(r.content)
    print('Downloading Zip file: ', str(data_path))


def unzip_data():
    """
    Unzip data that was downloaded
    :return:
    """
    assert data_path.is_file(), 'You need to double check the download code'
    zip_ref = zipfile.ZipFile(data_path, 'r')
    zip_ref.extractall(data_path.parent)
    zip_ref.close()
    print('Unzipped file: ', str(data_path))


def make_dataframe():
    """
    Create dataframe from tsv file
    :return:
    """
    assert file_path.is_file(), 'You need to double check the unzipping code'
    df = pd.read_csv(file_path, sep='\t', names=['label', 'text'])
    return df


def master_data_handler():
    if not data_path.is_file():
        download_data()

    if not file_path.is_file():
        unzip_data()


if __name__ == '__main__':
    master_data_handler()
