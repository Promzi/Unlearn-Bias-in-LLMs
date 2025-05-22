# 📦 Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch, json, random, os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import time

# ======================
# ⚙️ Setup
# ======================

start_time = time.time()
print("🔵 Starting Bias Mitigation Pipeline...")

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

SAVE_DIR = Path("bias_mitigation_outputs")
HEATMAP_SAVE_DIR = SAVE_DIR / "heatmaps"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Found {len(FILES)} data files:")
for file in FILES:
    print(f"  - {file}")



# ======================
# 🧠 Load Model and Tokenizer
# ======================
print("🔵 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device_map="auto")
device = model.device
model.eval()

# ====================
# 🧹 Load and Sample BBQ Examples
# ====================
# print("🔵 Loading BBQ dataset...")
# all_examples = []
# FILES = list(DATA_DIR.glob("*.jsonl"))
# for file in FILES:
#     with open(file, "r") as f:
#         all_examples.extend([json.loads(line) for line in f])

# print(f"✅ Loaded {len(all_examples)} BBQ examples!")


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
balanced_examples = []
SAMPLES_PER_CATEGORY = 1

for category, examples in examples_by_category.items():
    if len(examples) <= SAMPLES_PER_CATEGORY:
        balanced_examples.extend(examples)
    else:
        balanced_examples.extend(random.sample(examples, SAMPLES_PER_CATEGORY))

print(f"\n🟢 Using {len(balanced_examples)} balanced examples for evaluation.")



# ======================
# 📏 Measure Bias Score
# ======================
def measure_bias_score(model, examples):
    print("🔵 Measuring bias score...")
    model.eval()
    correct = 0
    total = 0

    for idx, example in enumerate(examples):
        if idx % 500 == 0:
            print(f"   ➔ Processing example {idx}/{len(examples)}...")

        context = example["context"]
        question = example["question"]
        options = [example["ans0"], example["ans1"], example["ans2"]]
        gold_idx = example["label"]  # 0,1,2

        # Shuffle options
        option_indices = list(range(3))
        random.shuffle(option_indices)
        shuffled_options = [options[i] for i in option_indices]
        new_gold_idx = option_indices.index(gold_idx)

        prompt = f"{context} {question}\n"
        for i, opt in enumerate(shuffled_options):
            prompt += f"{chr(65+i)}. {opt}\n"
        prompt += "\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

        probs = F.softmax(logits, dim=-1)

        a_id = tokenizer.encode(" A", add_special_tokens=False)[0]
        b_id = tokenizer.encode(" B", add_special_tokens=False)[0]
        c_id = tokenizer.encode(" C", add_special_tokens=False)[0]

        pred_idx = torch.argmax(torch.tensor([probs[a_id], probs[b_id], probs[c_id]]))

        if pred_idx.item() == new_gold_idx:
            correct += 1
        total += 1

    bias_score = 1 - (correct / total)
    print(f"✅ Bias Score Computed: {bias_score:.4f}")
    return bias_score

# ======================
# 🎯 Compute Attention-Context Correlation
# ======================
def compute_head_correlations(model, examples):
    print("🔵 Computing attention-context correlation...")
    results = {}
    model.eval()

    for h in range(model.config.num_attention_heads):
        activations = []
        influences = []

        for idx, example in enumerate(examples):
            if idx % 500 == 0:
                print(f"   ➔ [Head {h}] Processing {idx}/{len(examples)}...")

            context = example["context"]
            question = example["question"]
            options = [example["ans0"], example["ans1"], example["ans2"]]

            full_prompt = f"{context} {question}\n"
            for i, opt in enumerate(options):
                full_prompt += f"{chr(65+i)}. {opt}\n"
            full_prompt += "\nAnswer:"

            question_only_prompt = f"{question}\n"
            for i, opt in enumerate(options):
                question_only_prompt += f"{chr(65+i)}. {opt}\n"
            question_only_prompt += "\nAnswer:"

            # ---- Full prompt ----
            full_inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                full_outputs = model(**full_inputs, output_attentions=True)
                full_logits = full_outputs.logits[0, -1, :]
                attn = full_outputs.attentions[-1][0, h]

            # ---- Question-only prompt ----
            question_only_inputs = tokenizer(question_only_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                question_only_outputs = model(**question_only_inputs)
                question_only_logits = question_only_outputs.logits[0, -1, :]

            # ---- Probabilities ----
            full_probs = torch.softmax(full_logits, dim=-1)
            question_only_probs = torch.softmax(question_only_logits, dim=-1)

            a_id = tokenizer.encode(" A", add_special_tokens=False)[0]
            b_id = tokenizer.encode(" B", add_special_tokens=False)[0]
            c_id = tokenizer.encode(" C", add_special_tokens=False)[0]

            # ---- Compute Context Influence ----
            prob_a_full = full_probs[a_id].item()
            prob_b_full = full_probs[b_id].item()
            prob_c_full = full_probs[c_id].item()

            prob_a_q = question_only_probs[a_id].item()
            prob_b_q = question_only_probs[b_id].item()
            prob_c_q = question_only_probs[c_id].item()

            diff_a = np.log(max(prob_a_full, 1e-10)) - np.log(max(prob_a_q, 1e-10))
            diff_b = np.log(max(prob_b_full, 1e-10)) - np.log(max(prob_b_q, 1e-10))
            diff_c = np.log(max(prob_c_full, 1e-10)) - np.log(max(prob_c_q, 1e-10))

            context_influence = prob_a_full * diff_a + prob_b_full * diff_b + prob_c_full * diff_c

            # ---- Record
            max_attn = attn.max().item()
            activations.append(max_attn)
            influences.append(context_influence)

        # Correlation per head
        corr = np.corrcoef(activations, influences)[0, 1]
        results[h] = corr

    print("✅ Completed correlation computation.")
    return results

# ======================
# ✂️ Prune Heads Based on Correlation
# ======================
def prune_heads(model, head_correlations, thresholds=(0.005, 0.01, 0.02, 0.05)): #thresholds=(0.1, 0.3, 0.5, 0.7)
    print("🔵 Pruning heads based on correlation...")
    model.train()
    # device = model.device
    pruned_heads = {}
    

    for layer_idx, block in enumerate(model.model.layers):
        attn = block.self_attn
        heads_to_prune = []

        for h in range(model.config.num_attention_heads):
            corr = abs(head_correlations.get(h, 0))
            if corr >= thresholds[3]:
                strength = 1.0
            elif corr >= thresholds[2]:
                strength = 0.5
            elif corr >= thresholds[1]:
                strength = 0.3
            elif corr >= thresholds[0]:
                strength = 0.1
            else:
                strength = 0.0

            if strength > 0:
                with torch.no_grad():
                    q_proj = attn.q_proj.weight
                    k_proj = attn.k_proj.weight
                    v_proj = attn.v_proj.weight

                    start = h * attn.head_dim
                    end = (h+1) * attn.head_dim

                    q_proj[start:end] *= (1 - strength)
                    k_proj[start:end] *= (1 - strength)
                    v_proj[start:end] *= (1 - strength)

                heads_to_prune.append((h, strength))

        pruned_heads[layer_idx] = heads_to_prune

    print(f"✅ Finished pruning heads. Total layers pruned: {len(pruned_heads)}")
    return pruned_heads

# ======================
# 🔧 Fine-tuning Step
# ======================
def fine_tune(model, examples, epochs=2, lr=2e-5):
    print("🔵 Starting fine-tuning...")
    model.train()
    # device = model.device
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch+1}/{epochs}...")
        for idx, ex in enumerate(examples):
            if idx % 1000 == 0:
                print(f"   ➔ Fine-tuning on example {idx}/{len(examples)}")

            context = ex["context"]
            question = ex["question"]
            options = [ex["ans0"], ex["ans1"], ex["ans2"]]
            prompt = f"{context} {question}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            labels = inputs.input_ids.clone()

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(examples)
        print(f"✅ Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

# ======================
# 🚀 Main Experiment
# ======================

print("Evaluating bias before pruning...")
# bias_before = measure_bias_score(model, all_examples)
bias_before = measure_bias_score(model, balanced_examples)
print(f"Bias Score (Before): {bias_before:.4f}")

print("Computing head correlations...")
# head_corrs = compute_head_correlations(model, all_examples)
head_corrs = compute_head_correlations(model, balanced_examples)

print("Pruning based on correlation...")
pruned_heads = prune_heads(model, head_corrs)
print("Pruned heads:", pruned_heads)

print("Fine-tuning model after pruning...")
# fine_tune(model, all_examples, epochs=2)
fine_tune(model, balanced_examples, epochs=2)

print("Evaluating bias after pruning + fine-tuning...")
# bias_after = measure_bias_score(model, all_examples)
bias_after = measure_bias_score(model, balanced_examples)
print(f"Bias Score (After): {bias_after:.4f}")

# ======================
# 📊 Save Results
# ======================
results = {
    "bias_before": bias_before,
    "bias_after": bias_after,
    "pruned_heads": pruned_heads,
}

pd.DataFrame([results]).to_csv(SAVE_DIR / "unlearning_results.csv", index=False)

print("✅ Done!")