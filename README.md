# Unlearning for Fairness in Large Language Model
This repository contains a full pipeline for detecting, localizing, and mitigating social bias in large language models using the BBQ dataset. The pipeline is tailored for the meta-llama/Llama-3.2-1B-Instruct model, but can be adapted to similar decoder-only architectures.

# 📌 Features
> Evaluate social bias by category (e.g., race, gender, disability).
> Visualize hidden states, attention maps, and MLP activations.
> Identify attention heads correlated with context sensitivity.
> Apply soft pruning to mitigate biased heads.
> Fine-tune model post-pruning to preserve fluency and reduce hallucinations.
> Visualize model behavior before and after bias mitigation.

# 🧠 Project Structure
├── BBQ/data/                        # JSONL-format input data for all categories 

├── BBQ_llama3_2_1B_50sam_GTsoft/    # Output directory with evaluation & mitigation results

│   ├── bias_evaluation/

│   ├── bias_localization_before/

│   ├── bias_mitigation/

│   └── bias_localization_after/

└── main.py (this script)

# 📐 Setup
1. Dependencies:
    > Python ≥ 3.8
    > PyTorch with GPU + CUDA
    > Transformers (HuggingFace)
    > Seaborn, Matplotlib, NumPy, Pandas
    
pip install torch transformers matplotlib seaborn pandas

2. Login to Hugging Face Hub:
Update the line in the script with your personal token:

login(token="your_huggingface_token")

3. Model & Data:
  > Model: meta-llama/Llama-3.2-1B-Instruct (downloads automatically)
  > Dataset: Place .jsonl files from BBQ in BBQ/data/.

# 🚀 Running the Pipeline
1. Bias Evaluation
Evaluates the model's accuracy over:
  > Ambiguous context
  > Disambiguated context
  > Computes bias scores across identity categories.

Outputs:
  > bias_scores.txt in bias_evaluation/

2. Bias Localization
Captures and saves:
  > Hidden states (last layer)
  > MLP outputs
  > Attention maps (all heads)

Outputs:
  > Heatmaps saved in bias_localization_before/

3. Bias Mitigation (Unlearning)
  > Computes Pearson correlation between attention heads and context sensitivity.
  > Soft-prunes biased heads based on thresholds.
  > Fine-tunes model on disambiguated samples to restore coherence.
  > Evaluates the model again.

Outputs:
  > bias_scores_after.txt
  > unlearning_results.csv
  > Visuals in bias_localization_after/
