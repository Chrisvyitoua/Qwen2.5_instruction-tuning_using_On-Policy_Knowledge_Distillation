# Qwen2.5 Instruction-Tuning via On-Policy Knowledge Distillation

Distilling instruction-following capabilities from Qwen2.5-1.5B-Instruct into Qwen2.5-0.5B using Generalized Knowledge Distillation (GKD).

## Overview

**Teacher:** `Qwen/Qwen2.5-1.5B-Instruct` (1.5B params)
**Student:** `Qwen/Qwen2.5-0.5B` (0.5B params)
**Technique:** GKD (Generalized Knowledge Distillation) - On-Policy 
**Dataset:** `yahma/alpaca-cleaned` 
**Framework:** HuggingFace TRL (`GKDTrainer`) 

## Technique: On-Policy Knowledge Distillation

Unlike standard (off-policy) distillation where the student learns from fixed data, **GKD uses on-policy learning**:

1. Student **generates** its own response to a prompt
2. Teacher **evaluates** the student's generation
3. Student **learns** from teacher's feedback on its own outputs

**Reference:** [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649) (Agarwal et al., Google DeepMind, 2024)

## Key Findings

- **Behavior transfer:** Qwen2.5-0.5B successfully learned the instruction-response format.
- **Cleaner outputs:** Distilled model produces more focused, concise responses as compared to base model.

## Repository Structure
```
├── distilled_qwen_0.5b_instruct.ipynb   # Training notebook              # Model comparison notebook
└── README.md
```

## How to Run

1. Open `distilled_qwen_0.5b_instruct.ipynb` in Google Colab
2. Change runtime: **Runtime → Change runtime type → A100 GPU**
3. Run all cells

**Estimated training time:** ~2-6+ hours on A100 (depending on settings)
