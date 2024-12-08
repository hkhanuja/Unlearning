# AltPO

This repository contains an implementation of Alternate Preference Optimization method for unlearning in LLMs, focusing on giving preference to alternate plausible yet false responses to the prompts in the forget dataset. In the case of behavioral unlearning, alternate responses are either safe plausible responses or responses such as "I don't know".

## Installation

```bash
# Clone the repository
git clone https://github.com/Nikita-A-Tatarinov/Unlearning/tree/part_1_altpo

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

# Load or download a unlearned model
tokenizer = AutoTokenizer.from_pretrained("hkhanuja3/finetuned_alpaca_v0.1")
model = AutoModelForCausalLM.from_pretrained("hkhanuja3/finetuned_alpaca_v0.1")

# Generate text
question_start_token = "[INST] "
question_end_token = " [/INST]"
alpaca_input = question_start_token + prompt + question_end_token

alpaca_input_ids = tokenizer.encode(alpaca_input, return_tensors='pt').to(model.device)
alpaca_output_ids = model.generate(alpaca_input_ids, max_new_tokens=236)[0].detach()

alpaca_output = str(tokenizer.decode(alpaca_output_ids, skip_special_tokens=True))
```

## Training

To train the unlearning model:

```bash
Run the 
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


## Model Architecture

The unlearning system consists of:
1. Base Model: Original LLM 
2. Assistant Model: Smaller model trained to focus on target knowledge
3. Logit Subtraction: Combines outputs to remove specific knowledge
