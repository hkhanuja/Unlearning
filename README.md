# Part 3: Unsafe-to-Safe Response Transformation Framework

This part of the project focuses on transforming unsafe responses from LLMs into safe alternatives. The framework combines a classification step to detect unsafe responses with a Seq2Seq model to generate safe replacements. It operates on data outputs from various unlearning methods and produces safer, more acceptable prompt-response pairs.

---

## Overview

The transformation pipeline involves two main components:
1. **LSTM Classifier**: Identifies unsafe responses based on predefined categories (e.g., Privacy Violation, Harmful Content).
2. **Seq2Seq Transformer**: Converts unsafe responses into safe alternatives.

This part operates on data from `./Methods_Data/` and generates transformed responses and evaluation metrics in `./Final_Data/`.

---

## Code Structure

```
├── Methods_Data/             # Prompt-response data after unlearning methods
├── Final_Data/               # Transformed responses and evaluation metrics
├── SafeRLHF/
│   ├── data/                 # Dataset folder
│   |   ├── classification/   # Preprocessed data for classification training
│   |   ├── transformation/   # Preprocessed data for Seq2Seq transformation
│   |   └── original.csv      # Original dataset of prompt-response pairs
│   ├── models/               # Pre-trained models in .pth format
│   |   ├── classification/   # LSTM classifier models
│   |   └── transformation/   # Seq2Seq transformer models
│   ├── training/             # Training scripts for classification and transformation models
│   │   ├── classification.py # Implements the LSTM-based classifier
│   │   └── transformation.py # Implements the Seq2Seq transformation model
│   └── config.py             # Hyperparameters and training configurations
├── activate.sh               # Shell script to activate the environment
├── config.py                 # Configurations for logging, directories, and data files
├── environment.yml           # Conda environment configuration
├── get_stats.py              # Utility to compute and export metrics like %Safe and Acceptability Scores
├── requirements.txt          # Python dependencies for pip users
├── torch_requirements.txt    # Additional PyTorch-specific dependencies
├── transform.py              # Main script for running the transformation pipeline
└── update.sh                 # Shell script to update the environment
```

---

## How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/Nikita-A-Tatarinov/Unlearning.git
cd Unlearning
git checkout -b part_3_transformation origin/part_3_transformation
```

### 2. Set Up the Environment
```bash
source activate.sh
```

### 3. Run the Transformation Pipeline and Getting Stats
```bash
python transform.py
python get_stats.py
```

### 4. View Results
- Transformed prompt-response pairs: ./Final_Data/
- Comparison statistics (e.g., %Safe, Acceptability Scores): ./Final_Data/

