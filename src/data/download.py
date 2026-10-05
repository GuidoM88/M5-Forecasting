"""Download after accepting M5 competition rules and configuring Kaggle credentials."""
import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


def download_m5_dataset(data_dir='data/raw'):
    if shutil.which('kaggle') is None:
        raise RuntimeError('Install the download extra: pip install -e ".[download]"')
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(['kaggle', 'competitions', 'download', '-c', 'm5-forecasting-accuracy',
                    '-p', str(data_dir)], check=True)
    archive = data_dir / 'm5-forecasting-accuracy.zip'
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if not (data_dir / member.filename).resolve().is_relative_to(data_dir):
                raise ValueError('Unsafe archive path')
        zipped.extractall(data_dir)
    print(f'Dataset extracted to {data_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/raw')
    download_m5_dataset(parser.parse_args().output)
