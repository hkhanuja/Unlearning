# Behavioral Unlearning in Large Language Models

This repository contains the codebase for our project, **Behavioral Unlearning in Large Language Models**, which explores methods to make LLMs safer by selectively removing unsafe behavioral knowledge. The project is divided into three main parts, with each method implemented in a separate Git branch for modularity and clarity.

## Project Overview

### Part 1: Evaluating Existing Unlearning Methods
In this phase, we evaluate various existing unlearning methods for their effectiveness in removing unsafe knowledge while preserving the general utility of the model. Each unlearning method is implemented in a dedicated branch:
- **Branch `part_1_altpo`**: Implements the Alternate Preference Optimization (AltPO) method.
- **Branch `part_1_smpo`**: Implements the Simplicity Prevails (SimPO) method.
- **Branch `part_1_rmu`**: Implements the Representation Misdirection for Unlearning (RMU) method.
- **Branch `part_1_rfro`**: Implements the Reversing the Forget-Retain Objectives (RFRO) method.
- **Branch `part_1_sku`**: Implements the Selective Knowledge Negation Unlearning (SKU) method.

### Part 2: Prompt Engineering
This phase involves designing adversarial prompts to test the robustness of unlearning methods. We explore techniques to "jailbreak" unlearned models and extract unsafe responses. The corresponding implementation is in:
- **Branch `part_2_jailbreaking`**

### Part 3: Unsafe-to-Safe Response Transformation Framework
In this phase, we developed a lightweight framework to transform unsafe responses into safe ones. The framework uses:
1. An **LSTM-based classifier** to detect unsafe responses.
2. A **Transformer-based Seq2Seq model** to transform unsafe responses into safer alternatives.

This phase is implemented in:
- **Branch `data_and_models`: Contains code for preprocessing data to support all unlearning method experiments and to train classification and Seq2Seq models.**
- **Branch `part_3_transformation`: Implements the unsafe-to-safe response transformation pipeline.**
