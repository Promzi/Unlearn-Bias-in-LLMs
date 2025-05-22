# summarizing the heahmaps to see all samples together

# ======================
# 📦 Imports
# ======================
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch, json, random, os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import numpy as np
import torch.nn.functional as F

# ======================
# ⚙️ Setup
# ======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

login(token="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")

CACHE_DIR = "/scratch/phossai/model_cache"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATA_DIR = Path("BBQ/data")
FILES = list(DATA_DIR.glob("*.jsonl"))

SAVE_DIR = Path("bias_localization_outputs_test")
HEATMAP_SAVE_DIR = SAVE_DIR / "heatmaps"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Found {len(FILES)} data files:")
for file in FILES:
    print(f"  - {file}")

# ======================
# 🧠 Load Model and Tokenizer
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device_map="auto")
model.eval()

# ======================
# 🧹 Load and Sample BBQ Examples
# ======================
all_examples = []
examples_per_file = {}

for file in FILES:
    category = file.stem.split('.')[0]
    with open(file, "r") as f:
        file_data = [json.loads(line) for line in f]
        for ex in file_data:
            ex["source_file"] = str(file)
            ex["category_type"] = category
        all_examples.extend(file_data)
        examples_per_file[str(file)] = len(file_data)
        print(f"Loaded {len(file_data)} examples from {file}")

# Group by category
examples_by_category = {}
for ex in all_examples:
    category = ex["category"]
    examples_by_category.setdefault(category, []).append(ex)

# Balance sampling: 5 per category
# balanced_examples = []
# SAMPLES_PER_CATEGORY = 1

# for category, examples in examples_by_category.items():
#     if len(examples) <= SAMPLES_PER_CATEGORY:
#         balanced_examples.extend(examples)
#     else:
#         balanced_examples.extend(random.sample(examples, SAMPLES_PER_CATEGORY))

# print(f"\n🟢 Using {len(balanced_examples)} balanced examples for evaluation.")

# 🚨 Use all examples directly
balanced_examples = all_examples
print(f"\n🟢 Using {len(balanced_examples)} total examples for evaluation.")


# ==========================
# 📦 Summarization Functions
# ==========================

def summarize_hidden(hidden_tensor):
    """Summarize hidden state tensor"""
    hidden_mean = hidden_tensor.abs().mean().item()
    hidden_max = hidden_tensor.abs().max().item()
    hidden_sparsity = (hidden_tensor.abs() < 1e-5).float().mean().item()  # % of near-zero neurons
    return hidden_mean, hidden_max, hidden_sparsity

def summarize_attention(attn_tensor):
    """Summarize attention head 0"""
    attn_mean = attn_tensor.mean().item()
    attn_max = attn_tensor.max().item()
    return attn_mean, attn_max

def summarize_mlp(mlp_tensor):
    """Summarize MLP output tensor"""
    mlp_mean = mlp_tensor.abs().mean().item()
    mlp_max = mlp_tensor.abs().max().item()
    mlp_sparsity = (mlp_tensor.abs() < 1e-5).float().mean().item()
    return mlp_mean, mlp_max, mlp_sparsity


# ======================
# 📊 Hooks for Hidden States + MLPs
# ======================
hidden_states_per_layer = {}
mlp_outputs_per_layer = {}

def hook_hidden_states(module, input, output, layer_idx):
    if isinstance(output, tuple):
        output = output[0]  # Llama model outputs (hidden_states, ...)
    hidden_states_per_layer[layer_idx] = output.detach().cpu()

def hook_mlp_outputs(module, input, output, layer_idx):
    if isinstance(output, tuple):
        output = output[0]
    mlp_outputs_per_layer[layer_idx] = output.detach().cpu()

hidden_hooks = []
mlp_hooks = []

for layer_idx, block in enumerate(model.model.layers):
    h1 = block.register_forward_hook(lambda module, input, output, idx=layer_idx: hook_hidden_states(module, input, output, idx))
    hidden_hooks.append(h1)
    if hasattr(block, 'mlp'):
        h2 = block.mlp.register_forward_hook(lambda module, input, output, idx=layer_idx: hook_mlp_outputs(module, input, output, idx))
        mlp_hooks.append(h2)

print(f"✅ Registered hooks.")

# ======================
# 🔥 Utility: Plot Heatmap
# ======================
def plot_heatmap(tensor, title, path, xlabel="Hidden Size / Tokens", ylabel="Tokens"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(tensor, cmap='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ======================
# 🔥 Utility: Analyze Example (Bias Analysis + Localization)
# ======================
def analyze_example(model, tokenizer, example, idx, device="cuda"):
    question = example["question"]
    options = [example["ans0"], example["ans1"], example["ans2"]]
    context = example["context"]
    context_condition = example["context_condition"]
    
    shuffled_option_indices = list(range(len(options)))
    random.shuffle(shuffled_option_indices)
    shuffled_options = [options[i] for i in shuffled_option_indices]
    shuffled_to_original = {i: shuffled_option_indices[i] for i in range(len(shuffled_option_indices))}
    
    options_text = "\n".join([f"{chr(65+i)}. {shuffled_options[i]}" for i in range(len(shuffled_options))])
    option_mapping = {chr(65+i): options[shuffled_option_indices[i]] for i in range(len(options))}
    
    prompt = f"{context} {question}\n{options_text}\n\nAnswer:"
    question_prompt = f"{question}\n{options_text}\n\nAnswer:"
    
    a_id = tokenizer.encode(" A", add_special_tokens=False)[0]
    b_id = tokenizer.encode(" B", add_special_tokens=False)[0]
    c_id = tokenizer.encode(" C", add_special_tokens=False)[0]
    a_id_alt = tokenizer.encode("A", add_special_tokens=False)[0]
    b_id_alt = tokenizer.encode("B", add_special_tokens=False)[0]
    c_id_alt = tokenizer.encode("C", add_special_tokens=False)[0]

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(inputs["input_ids"])
    
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=attention_mask, output_attentions=True)
        logits = outputs.logits[0, -1, :]
        attentions = outputs.attentions

    probs = F.softmax(logits, dim=-1)

    prob_a_full = float(probs[a_id] + probs[a_id_alt])
    prob_b_full = float(probs[b_id] + probs[b_id_alt])
    prob_c_full = float(probs[c_id] + probs[c_id_alt])
    sum_full = prob_a_full + prob_b_full + prob_c_full
    if sum_full > 0:
        prob_a_full /= sum_full
        prob_b_full /= sum_full
        prob_c_full /= sum_full

    inputs_q = tokenizer(question_prompt, return_tensors="pt").to(device)
    attention_mask_q = torch.ones_like(inputs_q["input_ids"])
    
    with torch.no_grad():
        outputs_q = model(input_ids=inputs_q["input_ids"], attention_mask=attention_mask_q)
        logits_q = outputs_q.logits[0, -1, :]

    probs_q = F.softmax(logits_q, dim=-1)
    prob_a_q = float(probs_q[a_id] + probs_q[a_id_alt])
    prob_b_q = float(probs_q[b_id] + probs_q[b_id_alt])
    prob_c_q = float(probs_q[c_id] + probs_q[c_id_alt])
    sum_q = prob_a_q + prob_b_q + prob_c_q
    if sum_q > 0:
        prob_a_q /= sum_q
        prob_b_q /= sum_q
        prob_c_q /= sum_q

    diff_a = np.log(max(prob_a_full, 1e-10)) - np.log(max(prob_a_q, 1e-10))
    diff_b = np.log(max(prob_b_full, 1e-10)) - np.log(max(prob_b_q, 1e-10))
    diff_c = np.log(max(prob_c_full, 1e-10)) - np.log(max(prob_c_q, 1e-10))

    context_influence = prob_a_full * diff_a + prob_b_full * diff_b + prob_c_full * diff_c

    with torch.no_grad():
        outputs_gen = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
    
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = outputs_gen[0][prompt_length:]
    decoded_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    match = re.search(r'\b([ABC])\b', decoded_answer, re.IGNORECASE)
    response_letter = match.group(1).upper() if match else ""
    
    if response_letter:
        response_position = ord(response_letter) - ord('A')
        original_option_idx = shuffled_to_original.get(response_position, -1)
        original_answer = chr(65 + original_option_idx) if 0 <= original_option_idx < 3 else ""
    else:
        original_option_idx = -1
        original_answer = ""

    # 🔥 Save Heatmaps (only last layer)
    save_dir = HEATMAP_SAVE_DIR / f"example_{idx}"
    save_dir.mkdir(parents=True, exist_ok=True)

    hidden_mean = hidden_max = hidden_sparsity = None
    attn_mean = attn_max = None
    mlp_mean = mlp_max = mlp_sparsity = None

    if hidden_states_per_layer:
        last_hidden_idx = max(hidden_states_per_layer.keys())
        hs = hidden_states_per_layer[last_hidden_idx].squeeze(0)
        plot_heatmap(hs, "Hidden State Last Layer", save_dir / f"hidden_last.png")
        hidden_mean, hidden_max, hidden_sparsity = summarize_hidden(hs)

    if attentions:
        last_attn_idx = len(attentions) - 1
        attn = attentions[last_attn_idx][0,0].cpu()
        plot_heatmap(attn, "Attention Head 0 Last Layer", save_dir / f"attention_last.png")
        attn_mean, attn_max = summarize_attention(attn)

    if mlp_outputs_per_layer:
        last_mlp_idx = max(mlp_outputs_per_layer.keys())
        mlp = mlp_outputs_per_layer[last_mlp_idx].squeeze(0)
        plot_heatmap(mlp, "MLP Output Last Layer", save_dir / f"mlp_last.png")
        mlp_mean, mlp_max, mlp_sparsity = summarize_mlp(mlp)


    return {
        "example_id": example["example_id"],
        "category": example["category"],
        "context_condition": context_condition,
        "prompt": prompt,
        "model_response": response_letter,
        "original_option_picked": original_answer,
        "prob_A": prob_a_full,
        "prob_B": prob_b_full,
        "prob_C": prob_c_full,
        "prob_A_q": prob_a_q,
        "prob_B_q": prob_b_q,
        "prob_C_q": prob_c_q,
        "label": example["label"],
        "context_influence": context_influence,

        # ➡️ New Summary Metrics
        "hidden_mean": hidden_mean,
        "hidden_max": hidden_max,
        "hidden_sparsity": hidden_sparsity,
        "attn_mean": attn_mean,
        "attn_max": attn_max,
        "mlp_mean": mlp_mean,
        "mlp_max": mlp_max,
        "mlp_sparsity": mlp_sparsity
    }

# ======================
# 🚀 Main Evaluation Loop
# ======================
results = []
device = model.device

for idx, example in enumerate(balanced_examples):
    print(f"Processing example {idx+1}/{len(balanced_examples)}...")
    result = analyze_example(model, tokenizer, example, idx, device)
    results.append(result)

# Save results
df = pd.DataFrame(results)
df.to_csv(SAVE_DIR / "bias_localization_results.csv", index=False)
print("\n✅ Results saved to bias_localization_outputs/bias_localization_results.csv")

# Cleanup
for h in hidden_hooks + mlp_hooks:
    h.remove()

print("\n✅ Hooks removed. DONE!")


#### Summarization Anlayis of Heatmaps, Attention, MLP #####

# Load saved CSV
results_df = pd.read_csv(SAVE_DIR / "bias_localization_results.csv")

# --- 1. Hidden Mean Activation vs. Context Influence
plt.figure(figsize=(8,6))
sns.scatterplot(data=results_df, x="hidden_mean", y="context_influence", hue="category", palette="tab10")
plt.title("Hidden Mean Activation vs Context Influence")
plt.xlabel("Mean Hidden Activation (Last Layer)")
plt.ylabel("Context Influence Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(SAVE_DIR / "hidden_mean_vs_context_influence.png")
plt.show()

# --- 2. Attention Max vs. Context Influence
plt.figure(figsize=(8,6))
sns.scatterplot(data=results_df, x="attn_max", y="context_influence", hue="category", palette="tab10")
plt.title("Attention Max (Head 0) vs Context Influence")
plt.xlabel("Max Attention Score (Last Layer, Head 0)")
plt.ylabel("Context Influence Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(SAVE_DIR / "attention_max_vs_context_influence.png")
plt.show()

# --- 3. MLP Sparsity vs. Context Influence
plt.figure(figsize=(8,6))
sns.scatterplot(data=results_df, x="mlp_sparsity", y="context_influence", hue="category", palette="tab10")
plt.title("MLP Output Sparsity vs Context Influence")
plt.xlabel("MLP Output Sparsity (Fraction Near-Zero)")
plt.ylabel("Context Influence Score")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(SAVE_DIR / "mlp_sparsity_vs_context_influence.png")
plt.show()


##### Correlation Analysis #####


# Select only the numeric features we want to correlate
features_to_correlate = [
    "hidden_mean", "hidden_max", "hidden_sparsity",
    "attn_mean", "attn_max",
    "mlp_mean", "mlp_max", "mlp_sparsity",
    "context_influence"
]

# Drop rows where any of these features are NaN
results_df_corr = results_df[features_to_correlate].dropna()

# Compute Pearson correlation matrix
correlation_matrix = results_df_corr.corr(method="pearson")

# Save correlation matrix as CSV
correlation_matrix.to_csv(SAVE_DIR / "feature_bias_correlation_matrix.csv")

print("\n✅ Saved correlation matrix to feature_bias_correlation_matrix.csv")

# Plot Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix: Model Internals vs Context Influence")
plt.tight_layout()
plt.savefig(SAVE_DIR / "feature_bias_correlation_heatmap.png")
plt.show()
print("✅ Saved correlation heatmap to feature_bias_correlation_heatmap.png")
