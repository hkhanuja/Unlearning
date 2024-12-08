# Data and Models for Behavioral Unlearning

This branch contains scripts and data for preprocessing, training, and managing models used in the Behavioral Unlearning project. It includes tools to prepare data for classification and Seq2Seq tasks, train LSTM-based classifiers and Transformer-based Seq2Seq models, and log results during training.

---

## Overview

The **data_and_models** branch focuses on:
1. **Data Processing**: Prepares raw datasets for classification and transformation tasks.
2. **Model Training**: Implements and trains LSTM classifiers for detecting unsafe responses and Seq2Seq models for transforming unsafe responses into safe alternatives.
3. **Logging and Metrics**: Tracks training performance, logs key metrics, and saves pre-trained models for further use.

---

## Code Structure

```
├── SafeRLHF/
│   ├── data/                  # Dataset folder
│   |   ├── classification/    # Preprocessed data for classification training
│   |   ├── transformation/    # Preprocessed data for Seq2Seq transformation
│   |   └── original.csv       # Original dataset of prompt-response pairs
│   ├── log/                   # Logs of training processes
│   │   ├── classification.txt # Logs for classification model training
│   │   └── transformation.txt # Logs for Seq2Seq model training
│   ├── models/                # Pre-trained models in .pth format
│   |   ├── classification/    # LSTM classifier models
│   |   └── transformation/    # Seq2Seq transformer models
│   ├── processing/            # Data processing scripts
│   │   ├── download.py        # Downloads and processes raw datasets
│   │   ├── classification.py  # Processes data for classification models
│   │   └── transformation.py  # Processes data for Seq2Seq models
│   ├── training/              # Training scripts for classification and transformation models
│   │   ├── classification.py  # Implements the LSTM-based classifier
│   │   └── transformation.py  # Implements the Seq2Seq transformation model
│   └── config.py              # Hyperparameters and training configurations
├── activate.sh                # Shell script to activate the environment
├── config.py                  # Configurations for logging, directories, and data files
├── environment.yml            # Conda environment configuration
├── requirements.txt           # Python dependencies for pip users
├── torch_requirements.txt     # Additional PyTorch-specific dependencies
└── update.sh                  # Shell script to update the environment
```

---

## How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/Nikita-A-Tatarinov/Unlearning.git
cd Unlearning
git checkout -b data_and_models origin/data_and_models
```

### 2. Set Up the Environment
```bash
source activate.sh
```

### 3. Process Data
```bash
Copy code
python -m SafeRLHF.processing.download
python -m SafeRLHF.processing.classification
python -m SafeRLHF.processing.transformation
```

### 4. Train Models
```bash
python -m SafeRLHF.training.classification
python -m SafeRLHF.training.transformation
```

### 5. Logs and Results
- Training logs are saved in ./SafeRLHF/log/.
- Pre-trained models are saved in ./SafeRLHF/models/.

