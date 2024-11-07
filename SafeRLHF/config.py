import os

from config import SAFERLHF_DIR

DATA_DIR: str = os.path.join(SAFERLHF_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
ORIGINAL_DATA_FILE_NAME: str = 'original'
ORIGINAL_DATA_FILE: str = os.path.join(
    DATA_DIR, f'{ORIGINAL_DATA_FILE_NAME}.csv')

CLS_DATA_DIR: str = os.path.join(DATA_DIR, 'classification')
os.makedirs(CLS_DATA_DIR, exist_ok=True)
CLS_DATA_STATS_FILE_NAME: str = 'metadata'
CLS_DATA_STATS_FILE: str = os.path.join(
    CLS_DATA_DIR, f'{CLS_DATA_STATS_FILE_NAME}.csv')

CLS_SAFE_DATA_FILE_NAME: str = 'safe'
CLS_SAFE_DATA_FILE: str = os.path.join(
    CLS_DATA_DIR, f'{CLS_SAFE_DATA_FILE_NAME}.csv')

CLS_TRAIN_TEST_DATA_DIR: str = os.path.join(CLS_DATA_DIR, 'train_test')
os.makedirs(CLS_TRAIN_TEST_DATA_DIR, exist_ok=True)

SEQ2SEQ_DATA_DIR: str = os.path.join(DATA_DIR, 'transformation')
os.makedirs(SEQ2SEQ_DATA_DIR, exist_ok=True)

MODELS_DIR: str = os.path.join(SAFERLHF_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MAX_FEATURES: int = 10000
SEQUENCE_LENGTH: int = 250
EMBEDDING_DIM: int = 128
HIDDEN_DIM: int = 64
DROPOUT: float = 0.2
EPOCHS: int = 5
BATCH_SIZE: int = 32

CLS_MODELS_DIR: str = os.path.join(MODELS_DIR, 'classification')
os.makedirs(CLS_MODELS_DIR, exist_ok=True)
CLS_MODELS_PLOTS_DIR: str = os.path.join(CLS_MODELS_DIR, 'plots')
os.makedirs(CLS_MODELS_PLOTS_DIR, exist_ok=True)

SEQ2SEQ_MODELS_DIR: str = os.path.join(MODELS_DIR, 'seq2seq')
os.makedirs(SEQ2SEQ_MODELS_DIR, exist_ok=True)
SEQ2SEQ_MODELS_PLOTS_DIR: str = os.path.join(SEQ2SEQ_MODELS_DIR, 'plots')
os.makedirs(SEQ2SEQ_MODELS_PLOTS_DIR, exist_ok=True)
