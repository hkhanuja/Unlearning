import logging
import os

LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO

ROOT_DIR: str = '.' + os.sep

SAFERLHF_DIR: str = os.path.join(ROOT_DIR, 'SafeRLHF')

METHODS_DATA_DIR: str = os.path.join(ROOT_DIR, 'Methods_Data')

FINAL_DATA_DIR: str = os.path.join(ROOT_DIR, 'Final_Data')
os.makedirs(FINAL_DATA_DIR, exist_ok=True)
