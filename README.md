# Privacy Unlearning for Large Language Models

This repository contains an implementation of privacy unlearning for Large Language Models (LLMs), focusing on removing specific knowledge while preserving model utility. The implementation is based on logit subtraction between a base model and an assistant model trained to focus on the target knowledge.

## Installation

```bash
# Clone the repository
git clone https://github.com/Nikita-A-Tatarinov/Unlearning/tree/part_1_rfro

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from RFRO_wrapped import UnlearnedModelWrapper

# Load or download a pretrained model
model = UnlearnedModelWrapper.from_pretrained(
    download_model=True,  # Will download base model if not provided
    assistant_model_path="path/to/assistant/model"  # Path to unlearning weights
)

# Generate text
response = model.generate(
    "What is the capital of France?",
    max_length=512,
    temperature=0.7
)
```

## Training

To train the unlearning model:

```bash
python run.py
```

## Evaluation

To evaluate the unlearned model:

```bash
python evaluate.py
```

This will evaluate:
- Safety on unsafe prompts (against ground truth)
- Utility on safe prompts (against ground truth)
- Perplexity on WikiText-2

## Project Structure

```
privacy_unlearning/
├── data/
│   ├── Privacy Violation_train.csv
│   ├── Privacy Violation_test.csv
│   └── safe.csv
├── src/
│   ├── config.py            # Configuration classes
│   ├── model/
│   │   ├── assistant.py     # AssistantModel class
│   │   └── base.py         # BaseModel class
│   ├── tokenizer.py        # UnlearningTokenizer
│   ├── trainer.py          # UnlearningTrainer
│   ├── generator.py        # ResponseGenerator
│   └── unlearning.py       # Main AlpacaUnlearning class
├── utils/
│   ├── data_utils.py       # Data loading utilities
│   ├── logger.py           # Logging setup
│   └── model_utils.py      # Model downloading
├── checkpoints/            # Training checkpoints
├── outputs/               # Saved models
├── logs/                 # Log files
├── run.py               # Training script
└── evaluate.py          # Evaluation script
```

## Model Architecture

The unlearning system consists of:
1. Base Model: Original LLM 
2. Assistant Model: Smaller model trained to focus on target knowledge
3. Logit Subtraction: Combines outputs to remove specific knowledge