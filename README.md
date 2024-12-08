# AltPO

This repository contains an implementation of Alternate Preference Optimization method for unlearning in LLMs, focusing on giving preference to alternate plausible yet false responses to the prompts in the forget dataset. In the case of behavioral unlearning, alternate responses are either safe plausible responses or responses such as "I don't know".

## Installation

```bash
# Clone the repository
git clone -b part_1_altpo https://github.com/Nikita-A-Tatarinov/Unlearning

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

# Load or download a unlearned model

```bash
tokenizer = AutoTokenizer.from_pretrained("hkhanuja3/finetuned_alpaca_v0.1")
model = AutoModelForCausalLM.from_pretrained("hkhanuja3/finetuned_alpaca_v0.1")
```

# Generate text

```bash
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
Run the notebook forget_altpo_idk.ipynb
```

## Evaluation

To evaluate the unlearned model:

```bash
python evaluation/scoring.py
```

Run the 
```bash
evaluation/benchmark.ipynb
```

for perplexity evaluation on WikiTest-2 test data


This will evaluate:
- Percentage of unsafe responses to unsafe queries (should be lower for unlearned model)
- Percentage of safe responses to safe queries (should be higher or similar to base model for unlearned model)
- Perplexity on WikiText-2
