# Early Detection of Interest Loss in Hebrew WhatsApp Conversations

Final project for the **Natural Language Processing** course at  
**Reichman University, 2026**.

**Authors:**  
Roni Twito 

Matan Zohar Cohen 

## Overview

This project investigates whether loss of interest can be detected **before a conversation ends** in Hebrew WhatsApp-style sales conversations.

Rather than treating the task only as full-conversation classification, we study **early detection under partial observation** and examine whether behavioral communication patterns provide complementary information beyond textual content.

The project combines:

- Hebrew transformer-based text modeling using **AlephBERT**
- Behavioral and conversational features
- Prefix-aware training at 25%, 50%, 75%, and 100% of the conversation
- Text + behavioral feature fusion
- Controlled ablation experiments
- Bootstrap confidence intervals and McNemar statistical testing
- Class-level error analysis

## Dataset

The dataset contains **3,055 synthetic Hebrew WhatsApp-style sales conversations**.

All conversations, including the initial seed set, were synthetically generated specifically for this project.

No real WhatsApp or customer conversations were used.

Dataset split:

| Split | Conversations |
|---|---:|
| Train | 2,137 |
| Validation | 459 |
| Test | 459 |
| **Total** | **3,055** |

The two target classes are approximately balanced:

- `interested`
- `losing_interest`

## Experiments

### Full-Conversation Classification

| Model | Macro-F1 |
|---|---:|
| Pure Behavioral | 65.07% |
| Behavioral + Lexical | 86.69% |
| AlephBERT | 96.95% |
| AlephBERT + Behavioral Fusion | 97.38% |
| Continued AlephBERT (text-only control) | **98.04%** |

### Early Detection

Three experimental settings were evaluated:

**E1 — Full-conversation model evaluated on prefixes**  
The model is trained on complete conversations and evaluated after observing only 25%, 50%, 75%, or 100% of the conversation.

**E2 — Prefix-aware AlephBERT**  
AlephBERT is explicitly trained on partial conversation prefixes.

**E3 — Prefix-aware AlephBERT + Behavioral Fusion**  
Prefix-aware textual representations are combined with behavioral features calculated using only information available at the current prefix.

| Observed Conversation | E1 | E2 | E3 |
|---|---:|---:|---:|
| 25% | 33.28% | 62.64% | **64.45%** |
| 50% | 33.77% | 74.84% | **74.92%** |
| 75% | 51.14% | 86.93% | **89.11%** |
| 100% | **98.04%** | 95.20% | 94.32% |

## Key Findings

The main finding is that **strong full-conversation performance does not imply strong early-detection performance**.

Although the full-conversation model reaches **98.04% Macro-F1**, its performance drops substantially when only part of the conversation is available.

Prefix-aware training dramatically improves early detection. At 75% of the conversation:

**51.14% → 86.93% → 89.11%**

Full-conversation training → Prefix-aware text → Prefix-aware text + behavior.

Behavioral fusion provides modest and stage-dependent complementary information. At 75%, E3 improves over E2 by **+2.18 percentage points**.

Bootstrap 95% CI: **[+0.20, +4.36] pp**

McNemar: **p = 0.0639**

Therefore, we do **not** claim a statistically conclusive improvement from behavioral fusion.

## Repository Structure

```text
.
├── hebrew-sales-dataset-generator/   # Synthetic dataset generation
├── nlp-baseline/                     # Baseline experiments
├── text-model/                       # AlephBERT and early-detection experiments
│   ├── results/                      # Reproducible experiment results
│   └── tests/
│
└── report/
    ├── NLP_Final_Report.pdf
    ├── NLP_Final_Presentation.pdf
    └── latex_final/                  # LaTeX source and figures
