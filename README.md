# 🔍 Toward Inclusive Language Models
**Sparsity-Driven Calibration for Systematic and Interpretable Mitigation of Social Biases in LLMs**  
*EMNLP 2025 Submission*

This repository presents a full pipeline for detecting, localizing, and mitigating social biases in decoder-only transformer language models. Our method operates post hoc without requiring model retraining or gradient access. It is grounded in interpretable, architecture-aware interventions that systematically suppress stereotype amplification via sparsity-driven soft pruning.

---

## 📁 Repository Structure

├── BBQ.zip # Primary diagnostic dataset (bias benchmarking)

├── datasets.zip # Contains StereoSet, CrowS-Pairs, HellaSwag, MMLU

├── BiasAssessment_allsamples.py # Phase 1: Causal bias quantification (CI scoring)

├── BiasLocalization_layerwise... # Phase 2: Correlation-based component-level bias tracing

├── aya_8B.py # Full pipeline: Aya 8B model

├── llama3_2_1B.py # Full pipeline: LLaMA 3.2-1B model

├── llama3_2_3B.py # Full pipeline: LLaMA 3.2-3B model

├── llama3_1_8B.py # Full pipeline: LLaMA 3.1-8B model

├── qwen_32B.py # Full pipeline: Qwen 32B model

├── llama3_2_1B_catwisehead.py # Cross-dimensional attention head selection (Pearson + frequency)

├── phase3_working.py # Phase 3: Bias localization visualization

├── phase4_Real.py # Phase 4: Soft pruning & fine-tuning

├── llama_bbq_outputs_all.csv # Final model outputs and scores

├── layerwise_tracing_outputs.zip # Pre-extracted attention/MLP/hidden tracing outputs


---

## 🚀 Methodology Overview

Our framework is divided into four core stages:

### 1. **Causal Bias Quantification**
- File: `BiasAssessment_allsamples.py`
- Computes the **Context Influence (CI) Score** for each model prediction using ambiguous vs. disambiguated prompts.
- CI is a causal measure quantifying how context shifts the model's belief.

### 2. **Component-Level Bias Localization**
- File: `BiasLocalization_layerwise_tracing.py`
- Extracts activations (attention, hidden, MLP) and correlates them with CI.
- Identifies **bias-amplifying attention heads** using Pearson correlation.

### 3. **Cross-Dimensional Head Selection**
- File: `llama3_2_1B_catwisehead.py`
- Implements a principled head selection mechanism:
  - **High correlation magnitude**
  - **High cross-category consistency**
- Targets systemic bias across identity axes (gender, race, disability, etc.)

### 4. **Soft Pruning & Fine-Tuning**
- File: `phase4_Real.py`
- Attenuates Q/K/V matrices of implicated heads using calibrated α-scaling:
  
        $W_M^h ← (1 - α_h) * W_M^h, where M ∈ {Q, K, V}$

  - Applies lightweight fine-tuning to restore reasoning ability and fluency.

---

## 🧠 Model-Specific Pipelines

Each model has its own script that runs the entire pipeline:

| Script               | Model                  |
|----------------------|------------------------|
| `llama3_2_1B.py`     | LLaMA 3.2-1B           |
| `llama3_2_3B.py`     | LLaMA 3.2-3B           |
| `llama3_1_8B.py`     | LLaMA 3.1-8B           |
| `aya_8B.py`          | Aya 8B                 |
| `qwen_32B.py`        | Qwen 32B               |

Each script:
- Loads pretrained HuggingFace model/tokenizer
- Performs bias evaluation, localization, pruning, and post-eval
- Saves outputs to: `BBQ_<model_name>_<method_variant>/`

---

## 📂 Datasets Used

| Dataset        | Description                                                  | Purpose              |
|----------------|--------------------------------------------------------------|-----------------------|
| **BBQ**        | Bias Benchmark for QA across 9 demographics                 | Bias training & eval |
| **StereoSet**  | Stereotypical vs. anti-stereotypical completions            | Held-out test bias   |
| **CrowS-Pairs**| Sentence-pair counterfactual bias benchmark                 | Held-out validation  |
| **MMLU**       | 57-domain multiple choice reasoning benchmark               | Utility evaluation   |
| **HellaSwag**  | Commonsense reasoning for sentence completion               | Utility evaluation   |

---

## 📈 Output Folders (Example)

For `llama3_2_1B.py`, outputs are saved to:

BBQ_llama3_2_1B_50sam_GTsoft/

├── bias_evaluation/

├── bias_localization_before/

├── bias_localization_after/

├── bias_mitigation/

└── unlearning_results.csv


Each folder contains:
- Pre/post pruning attention maps
- CI scores
- Bias heatmaps
- Fine-tuned model performance

---

## 📌 Key Contributions

> As described in our paper (EMNLP 2025 submission), this repo supports:

1. **Context Influence (CI) Score** — a causal proxy for measuring stereotype sensitivity.
2. **Correlation-based Localization** — interpretable mapping of model bias to attention heads.
3. **Multi-criteria Head Selection** — detects systemic, cross-demographic bias pathways.
4. **Soft Pruning Strategy** — scalable mitigation without retraining.
5. **Fine-Tuning Calibration** — restores fluency and factual coherence post-pruning.

---

## ⚙️ Setup Instructions

### Dependencies

```bash
pip install torch transformers matplotlib seaborn pandas

## 🔐 Hugging Face Authentication

To download models like `meta-llama/Llama-3.2-1B-Instruct`, you need to authenticate with the Hugging Face Hub:

```python
from huggingface_hub import login
login(token="your_huggingface_token_here")

▶️ How to Run
🔁 Full Pipeline

```bash
python llama3_2_1B.py

---

### 🔥 COMMON MISTAKE:

If you forget to **close the code block**, like this:

```markdown
```python
from huggingface_hub import login
login(token="your_huggingface_token_here")
## 🔐 Hugging Face Authentication   <-- This becomes part of the Python block ❌
