import os

ROOT_DIR: str = '.' + os.sep

DATA_DIR: str = os.path.join(ROOT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
ORIGINAL_DATA_FILE: str = os.path.join(DATA_DIR, 'original.csv')

CLASSIFICATION_DATA_DIR: str = os.path.join(DATA_DIR, 'classification')
os.makedirs(CLASSIFICATION_DATA_DIR, exist_ok=True)
SAFE_DATA_FILE: str = os.path.join(CLASSIFICATION_DATA_DIR, 'safe.csv')

TRANSFORMATION_DATA_DIR: str = os.path.join(DATA_DIR, 'transformation')
os.makedirs(TRANSFORMATION_DATA_DIR, exist_ok=True)
