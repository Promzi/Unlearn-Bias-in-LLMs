# 📦 Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch, json, random, os, time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast


# =============================
# ⚙️ Setup
# =============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

login(token="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") # Login to Hugging Face Hub

CACHE_DIR = "/scratch/phossai/model_cache"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATA_DIR = Path("BBQ/data")
FILES = list(DATA_DIR.glob("*.jsonl"))

# Save Directories
BASE_SAVE_DIR = Path("BBQ_llama3_2_1B_catheadLW")
BIAS_EVAL_DIR = BASE_SAVE_DIR / "bias_evaluation"
BIAS_LOCALIZATION_BEFORE_DIR = BASE_SAVE_DIR / "bias_localization_before"
BIAS_MITIGATION_DIR = BASE_SAVE_DIR / "bias_mitigation"
BIAS_LOCALIZATION_AFTER_DIR = BIAS_MITIGATION_DIR / "heatmaps_after"

for d in [BIAS_EVAL_DIR, BIAS_LOCALIZATION_BEFORE_DIR, BIAS_MITIGATION_DIR, BIAS_LOCALIZATION_AFTER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Found {len(FILES)} data files:")
for file in FILES:
    print(f"  - {file}")

# =============================
# 🧠 Load Model and Tokenizer
# =============================
print("🔵 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device_map="auto")
device = model.device
model.eval()

# =============================
# 🧹 Load BBQ Examples
# =============================
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

# Sampling
balanced_examples = []
SAMPLES_PER_CATEGORY = 20
for category, examples in examples_by_category.items():
    if len(examples) <= SAMPLES_PER_CATEGORY:
        balanced_examples.extend(examples)
    else:
        balanced_examples.extend(random.sample(examples, SAMPLES_PER_CATEGORY))
print(f"\n🟢 Using {len(balanced_examples)} balanced examples for evaluation.")

# # 🚨 Use all examples directly
# balanced_examples = all_examples
# print(f"\n🟢 Using {len(balanced_examples)} total examples for evaluation.")

# =============================
# Step 1: Bias Evaluation
# =============================

def measure_bias_score_by_category(model, examples, categories):
    print(" Measuring bias scores by category...")
    model.eval()
    results = {"overall": {"correct": 0, "total": 0, "ambig_correct": 0, "ambig_total": 0, "disambig_correct": 0, "disambig_total": 0}}
    
    for cat in categories:
        results[cat] = {"correct": 0, "total": 0, "ambig_correct": 0, "ambig_total": 0, "disambig_correct": 0, "disambig_total": 0}
    
    for example in examples:
        context = example["context"]
        question = example["question"]
        options = [example["ans0"], example["ans1"], example["ans2"]]
        gold_idx = example["label"]
        context_condition = example["context_condition"]
        category = example["category"]

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

        correct = (pred_idx.item() == new_gold_idx)
        
        results["overall"]["total"] += 1
        results[category]["total"] += 1
        if context_condition == "ambig":
            results["overall"]["ambig_total"] += 1
            results[category]["ambig_total"] += 1
        else:
            results["overall"]["disambig_total"] += 1
            results[category]["disambig_total"] += 1

        if correct:
            results["overall"]["correct"] += 1
            results[category]["correct"] += 1
            if context_condition == "ambig":
                results["overall"]["ambig_correct"] += 1
                results[category]["ambig_correct"] += 1
            else:
                results["overall"]["disambig_correct"] += 1
                results[category]["disambig_correct"] += 1

    bias_report = "===== BIAS ANALYSIS =====\n"
    bias_report += "\nContext Influence by Condition:\n"
    bias_report += f"- ambig : {1 - results['overall']['ambig_correct']/max(1, results['overall']['ambig_total']):.4f}\n"
    bias_report += f"- disambig : {1 - results['overall']['disambig_correct']/max(1, results['overall']['disambig_total']):.4f}\n\n"

    bias_report += f"Overall Bias Score: {1 - results['overall']['correct']/results['overall']['total']:.4f} ( {results['overall']['correct']} / {results['overall']['total']} )\n"
    bias_report += f"- Ambig Bias Score: {1 - results['overall']['ambig_correct']/max(1, results['overall']['ambig_total']):.4f} ( {results['overall']['ambig_correct']} / {results['overall']['ambig_total']} )\n"
    bias_report += f"- Disambig Bias Score: {1 - results['overall']['disambig_correct']/max(1, results['overall']['disambig_total']):.4f} ( {results['overall']['disambig_correct']} / {results['overall']['disambig_total']} )\n"
    bias_report += "\nBias Metrics by Category:\n"

    for cat in categories:
        bias_report += f"\n{cat}:\n"
        bias_report += f"- Overall Bias Score: {1 - results[cat]['correct']/max(1, results[cat]['total']):.4f} ( {results[cat]['correct']} / {results[cat]['total']} samples )\n"
        bias_report += f"- Ambig Bias Score: {1 - results[cat]['ambig_correct']/max(1, results[cat]['ambig_total']):.4f} ( {results[cat]['ambig_correct']} / {results[cat]['ambig_total']} )\n"
        bias_report += f"- Disambig Bias Score: {1 - results[cat]['disambig_correct']/max(1, results[cat]['disambig_total']):.4f} ( {results[cat]['disambig_correct']} / {results[cat]['disambig_total']} )\n"

    return bias_report

categories = list(examples_by_category.keys())
start_time = time.time()
bias_report = measure_bias_score_by_category(model, balanced_examples, categories)
end_time = time.time()

inference_time_step1 = end_time - start_time

with open(BIAS_EVAL_DIR / "bias_scores.txt", "w") as f:
    f.write(bias_report)
    f.write(f"\nInference Time: {inference_time_step1:.2f} seconds\n")

print(bias_report)
print(f"\n✅ Bias Evaluation Completed in {end_time - start_time:.2f} seconds.")

# =============================
# Step 2: Bias Localization
# =============================

# Summarization helpers
def summarize_hidden(hidden_tensor):
    hidden_mean = hidden_tensor.abs().mean().item()
    hidden_max = hidden_tensor.abs().max().item()
    hidden_sparsity = (hidden_tensor.abs() < 1e-5).float().mean().item()
    return hidden_mean, hidden_max, hidden_sparsity

def summarize_attention(attn_tensor):
    attn_mean = attn_tensor.mean().item()
    attn_max = attn_tensor.max().item()
    return attn_mean, attn_max

def summarize_mlp(mlp_tensor):
    mlp_mean = mlp_tensor.abs().mean().item()
    mlp_max = mlp_tensor.abs().max().item()
    mlp_sparsity = (mlp_tensor.abs() < 1e-5).float().mean().item()
    return mlp_mean, mlp_max, mlp_sparsity

# Hook functions
hidden_states_per_layer = {}
mlp_outputs_per_layer = {}

def hook_hidden_states(module, input, output, layer_idx):
    if isinstance(output, tuple):
        output = output[0]
    hidden_states_per_layer[layer_idx] = output.detach().cpu()

def hook_mlp_outputs(module, input, output, layer_idx):
    if isinstance(output, tuple):
        output = output[0]
    mlp_outputs_per_layer[layer_idx] = output.detach().cpu()

# Register hooks
hidden_hooks = []
mlp_hooks = []

for layer_idx, block in enumerate(model.model.layers):
    h1 = block.register_forward_hook(lambda module, input, output, idx=layer_idx: hook_hidden_states(module, input, output, idx))
    hidden_hooks.append(h1)
    if hasattr(block, 'mlp'):
        h2 = block.mlp.register_forward_hook(lambda module, input, output, idx=layer_idx: hook_mlp_outputs(module, input, output, idx))
        mlp_hooks.append(h2)

print("✅ Registered hooks for Bias Localization.")

def plot_heatmap(tensor, title, path, xlabel="Tokens", ylabel="Hidden Size"):
    plt.figure(figsize=(10, 8), facecolor='white')  # white background
    ax = sns.heatmap(
        tensor,
        cmap='hot',
        cbar=True,
        square=True,
        linewidths=0,
        # linecolor='white',
        xticklabels=True,
        yticklabels=True
    )
    ax.set_facecolor("white")  # ensure heatmap area is also white
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, facecolor='white')  # enforce white bg in saved file
    plt.close()

# Analyze examples and save heatmaps
def analyze_example_and_save(model, tokenizer, example, idx, device="cuda"):
    question = example["question"]
    options = [example["ans0"], example["ans1"], example["ans2"]]
    context = example["context"]

    prompt = f"{context} {question}\n"
    for i, opt in enumerate(options):
        prompt += f"{chr(65+i)}. {opt}\n"
    prompt += "\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], output_attentions=True)
        attentions = outputs.attentions

    save_dir = BIAS_LOCALIZATION_BEFORE_DIR / f"example_{idx}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save Hidden State Heatmap
    if hidden_states_per_layer:
        hs = hidden_states_per_layer[max(hidden_states_per_layer.keys())].squeeze(0)  # shape [sequence_length, hidden_dim]
        # hs = hs.transpose(0, 1)  # [hidden_dim, sequence_length] → [tokens, hidden_size]
        plot_heatmap(hs, "Hidden State Last Layer", save_dir / "hidden_last.png", xlabel="Hidden Size", ylabel="Tokens")

    # Save MLP Heatmap
    if mlp_outputs_per_layer:
        mlp = mlp_outputs_per_layer[max(mlp_outputs_per_layer.keys())].squeeze(0)  # shape [sequence_length, hidden_dim]
        # mlp = mlp.transpose(0, 1)  # [hidden_dim, sequence_length] → [tokens, hidden_size]
        plot_heatmap(mlp, "MLP Last Layer", save_dir / "mlp_last.png", xlabel="Hidden Size", ylabel="Tokens")

    # Save Attention Heatmaps (all heads)
    if attentions:
        last_attn_idx = len(attentions) - 1
        attn_map = attentions[last_attn_idx][0]
        for head_idx in range(attn_map.shape[0]):
            attn_head = attn_map[head_idx].cpu()
            plot_heatmap(attn_head, f"Attention Head {head_idx}", save_dir / f"attention_head{head_idx}.png")

# Run over all balanced examples
for idx, example in enumerate(balanced_examples):
    print(f"Analyzing and saving heatmaps for example {idx+1}/{len(balanced_examples)}...")
    analyze_example_and_save(model, tokenizer, example, idx, device=device)

# Cleanup hooks after localization step
for h in hidden_hooks + mlp_hooks:
    h.remove()

print("✅ Finished Bias Localization heatmaps saving.")


# =============================
# Step 3: Bias Mitigation
# =============================

# 🎯 Compute Attention-Context Correlation
# def compute_head_correlations(model, examples):
#     print(" Computing attention-context correlation...")
#     results = {}
#     model.eval()

#     for h in range(model.config.num_attention_heads):
#         activations = []
#         influences = []

#         for example in examples:
#             context = example["context"]
#             question = example["question"]
#             options = [example["ans0"], example["ans1"], example["ans2"]]

#             prompt = f"{context} {question}\n"
#             for i, opt in enumerate(options):
#                 prompt += f"{chr(65+i)}. {opt}\n"
#             prompt += "\nAnswer:"

#             question_only_prompt = f"{question}\n"
#             for i, opt in enumerate(options):
#                 question_only_prompt += f"{chr(65+i)}. {opt}\n"
#             question_only_prompt += "\nAnswer:"

#             full_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#             with torch.no_grad():
#                 full_outputs = model(**full_inputs, output_attentions=True)
#                 full_logits = full_outputs.logits[0, -1, :]
#                 attn = full_outputs.attentions[-1][0, h]

#             question_only_inputs = tokenizer(question_only_prompt, return_tensors="pt").to(model.device)
#             with torch.no_grad():
#                 question_only_outputs = model(**question_only_inputs)
#                 question_only_logits = question_only_outputs.logits[0, -1, :]

#             full_probs = torch.softmax(full_logits, dim=-1)
#             question_only_probs = torch.softmax(question_only_logits, dim=-1)

#             a_id = tokenizer.encode(" A", add_special_tokens=False)[0]
#             b_id = tokenizer.encode(" B", add_special_tokens=False)[0]
#             c_id = tokenizer.encode(" C", add_special_tokens=False)[0]

#             prob_a_full = full_probs[a_id].item()
#             prob_b_full = full_probs[b_id].item()
#             prob_c_full = full_probs[c_id].item()

#             prob_a_q = question_only_probs[a_id].item()
#             prob_b_q = question_only_probs[b_id].item()
#             prob_c_q = question_only_probs[c_id].item()

#             diff_a = np.log(max(prob_a_full, 1e-10)) - np.log(max(prob_a_q, 1e-10))
#             diff_b = np.log(max(prob_b_full, 1e-10)) - np.log(max(prob_b_q, 1e-10))
#             diff_c = np.log(max(prob_c_full, 1e-10)) - np.log(max(prob_c_q, 1e-10))

#             context_influence = prob_a_full * diff_a + prob_b_full * diff_b + prob_c_full * diff_c

#             activations.append(attn.max().item())
#             influences.append(context_influence)

#         corr = np.corrcoef(activations, influences)[0, 1]
#         results[h] = corr

#     print("✅ Completed correlation computation.")
#     return results

# # 🔧 Prune Heads
# def prune_heads(model, head_correlations, thresholds=(0.05, 0.3, 0.5, 0.7)): #thresholds=(0.05, 0.1, 0.2, 0.5)
#     print("🔵 Pruning heads based on correlation...")
#     model.train()
#     pruned_heads = {}

#     for layer_idx, block in enumerate(model.model.layers):
#         attn = block.self_attn
#         heads_to_prune = []

#         for h in range(model.config.num_attention_heads):
#             corr = abs(head_correlations.get(h, 0))
#             if corr >= thresholds[3]:
#                 strength = 0.9
#             elif corr >= thresholds[2]:
#                 strength = 0.7
#             elif corr >= thresholds[1]:
#                 strength = 0.5
#             elif corr >= thresholds[0]:
#                 strength = 0.1
#             else:
#                 strength = 0.0

#             if strength > 0:
#                 with torch.no_grad():
#                     q_proj = attn.q_proj.weight
#                     k_proj = attn.k_proj.weight
#                     v_proj = attn.v_proj.weight

#                     start = h * attn.head_dim
#                     end = (h+1) * attn.head_dim

#                     q_proj[start:end] *= (1 - strength)
#                     k_proj[start:end] *= (1 - strength)
#                     v_proj[start:end] *= (1 - strength)

#                 heads_to_prune.append((h, strength))

#         pruned_heads[layer_idx] = heads_to_prune

#     print(f"✅ Finished pruning heads. Total layers pruned: {len(pruned_heads)}")
#     return pruned_heads

def compute_head_correlations_by_category(model, examples, tokenizer):
    print("🔎 Computing per-category head-context correlations...")
    model.eval()
    head_corrs_by_cat = {}

    for cat in set(ex["category"] for ex in examples):
        cat_examples = [ex for ex in examples if ex["category"] == cat]
        correlations = {}

        for layer_idx, block in enumerate(model.model.layers):
            for h in range(model.config.num_attention_heads):
                activations, influences = [], []

                for ex in cat_examples:
                    context = ex["context"]
                    question = ex["question"]
                    options = [ex["ans0"], ex["ans1"], ex["ans2"]]
                    full_prompt = f"{context} {question}\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) + "\nAnswer:"
                    qonly_prompt = f"{question}\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) + "\nAnswer:"

                    full_inp = tokenizer(full_prompt, return_tensors="pt").to(model.device)
                    qonly_inp = tokenizer(qonly_prompt, return_tensors="pt").to(model.device)

                    with torch.no_grad():
                        full_out = model(**full_inp, output_attentions=True)
                        qonly_out = model(**qonly_inp)

                    full_logits = full_out.logits[0, -1, :]
                    qonly_logits = qonly_out.logits[0, -1, :]

                    probs_f = torch.softmax(full_logits, dim=-1)
                    probs_q = torch.softmax(qonly_logits, dim=-1)

                    a_id = tokenizer.encode(" A", add_special_tokens=False)[0]
                    b_id = tokenizer.encode(" B", add_special_tokens=False)[0]
                    c_id = tokenizer.encode(" C", add_special_tokens=False)[0]

                    diffs = [np.log(max(probs_f[i].item(), 1e-10)) - np.log(max(probs_q[i].item(), 1e-10)) for i in [a_id, b_id, c_id]]
                    infl = sum(probs_f[i].item() * diffs[j] for j, i in enumerate([a_id, b_id, c_id]))

                    # attn = full_out.attentions[-1][0, h]
                    attn = full_out.attentions[layer_idx][0, h]  # [batch, head, seq, seq]
                    activations.append(attn.max().item())
                    influences.append(infl)

                corr = np.corrcoef(activations, influences)[0, 1] if len(activations) > 1 else 0.0
                # correlations[h] = corr
                correlations[(layer_idx, h)] = corr

        head_corrs_by_cat[cat] = correlations

    print("✅ Per-category correlation computed.")
    return head_corrs_by_cat


def prune_heads_per_category(model, head_corrs_by_cat, threshold=0.2, vote_thresh=0.3):
    print("✂️ Pruning heads based on category-specific correlation...")

    head_votes = {}
    head_weights ={}

    categories = list(head_corrs_by_cat.keys())
    num_cats = len(categories)

    for cat, head_corrs in head_corrs_by_cat.items():
        for (layer, h), corr in head_corrs.items():
            if abs(corr) >= threshold:
                # head_votes[h] = head_votes.get(h, 0) + 1
                key = (layer, h)
                head_votes[key] = head_votes.get(key, 0) + 1
                head_weights[key] = head_weights.get(key, 0) + abs(corr)  # accumulate weighted signal

    heads_to_prune = [h for h, votes in head_votes.items() if votes / num_cats >= vote_thresh]

    print(f"🧠 Pruning heads: {heads_to_prune}")

    for (layer_idx, head_idx) in heads_to_prune:
        attn = model.model.layers[layer_idx].self_attn
        with torch.no_grad():
            start = head_idx * attn.head_dim
            end = (head_idx + 1) * attn.head_dim
            attn.q_proj.weight[start:end] *= 0.0
            attn.k_proj.weight[start:end] *= 0.0
            attn.v_proj.weight[start:end] *= 0.0

    print(f"✅ Pruned {len(heads_to_prune)} heads across all layers.")
    return heads_to_prune, head_votes, head_weights

    # for layer_idx, block in enumerate(model.model.layers):
    #     attn = block.self_attn
    #     for h in heads_to_prune:
    #         with torch.no_grad():
    #             start = h * attn.head_dim
    #             end = (h + 1) * attn.head_dim
    #             attn.q_proj.weight[start:end] *= 0.0
    #             attn.k_proj.weight[start:end] *= 0.0
    #             attn.v_proj.weight[start:end] *= 0.0

    # print(f"✅ Pruned {len(heads_to_prune)} heads across all layers.")
    # return heads_to_prune, head_votes


def fine_tune(model, examples, epochs=2, lr=2e-5, accum_steps=4, max_len=512):
    print("🔵 Starting fine-tuning after pruning...")
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    for epoch in range(epochs):
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{epochs}...")
        optimizer.zero_grad()

        for idx, ex in enumerate(examples):
            if idx % 100 == 0:
                print(f"   ➔ Fine-tuning on example {idx}/{len(examples)}")

            # Build prompt
            context = ex["context"]
            question = ex["question"]
            options = [ex["ans0"], ex["ans1"], ex["ans2"]]
            prompt = f"{context} {question}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nAnswer:"

            # Tokenize with truncation
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
            labels = inputs.input_ids.clone()

            with autocast():
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss / accum_steps  # Normalize loss for gradient accumulation

            scaler.scale(loss).backward()
            total_loss += loss.item() * accum_steps  # Undo division for accurate tracking

            # Step optimizer after accum_steps
            if (idx + 1) % accum_steps == 0 or (idx + 1) == len(examples):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            # Optional: Delete tensors manually
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(examples)
        print(f"✅ Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")


# 🎯 Main Unlearning + After Evaluation

print("\n🔵 Computing head correlations...")
# head_corrs = compute_head_correlations(model, balanced_examples)
head_corrs_by_cat = compute_head_correlations_by_category(model, balanced_examples, tokenizer)


print("🔵 Pruning based on correlations...")
start_time_step3 = time.time()
# pruned_heads = prune_heads(model, head_corrs)
pruned_heads, head_votes, heads_weights = prune_heads_per_category(model, head_corrs_by_cat, threshold=0.2, vote_thresh=0.3)


print("🔵 Fine-tuning model after pruning...")
fine_tune(model, balanced_examples[:1000], epochs=1)
end_time_step3 = time.time()
mitigation_time = end_time_step3 - start_time_step3
print(f"✅ Unlearning completed in {mitigation_time:.2f} seconds.")


print("🔵 Evaluating bias after pruning and fine-tuning...")
bias_report_after = measure_bias_score_by_category(model, balanced_examples, categories)

with open(BIAS_MITIGATION_DIR / "bias_scores_after.txt", "w") as f:
    f.write(bias_report_after)
    f.write(f"\nMitigation Time: {mitigation_time:.2f} seconds\n")

print(bias_report_after)

# Save pruned heads record
# pd.DataFrame([{"layer": l, "pruned_heads": str(h)} for l,h in pruned_heads.items()]).to_csv(BIAS_MITIGATION_DIR / "unlearning_results.csv", index=False)
df_records = []
for (layer, head), vote_count in head_votes.items():
    pruned = (layer, head) in pruned_heads
    weight = head_weights.get((layer, head), 0.0)
    df_records.append({
        "layer": layer,
        "head": head,
        "vote_count": vote_count,
        "total_corr_weight": round(weight, 4),
        "pruned": pruned
    })

pd.DataFrame(df_records).to_csv(BIAS_MITIGATION_DIR / "unlearning_results.csv", index=False)

def plot_layerwise_bias_curve(head_corrs_by_cat, save_path):
    layer_count = max(l for (l, _) in next(iter(head_corrs_by_cat.values())).keys()) + 1
    category_curves = {}

    for cat, correlations in head_corrs_by_cat.items():
        layer_vals = []
        for l in range(layer_count):
            vals = [abs(v) for (layer, _), v in correlations.items() if layer == l]
            layer_vals.append(np.mean(vals) if vals else 0.0)
        category_curves[cat] = layer_vals

    plt.figure(figsize=(12, 6))
    for cat, vals in category_curves.items():
        highlight_start = next((i for i, v in enumerate(vals) if v > 0.05), len(vals))
        x = list(range(layer_count))
        plt.plot(x[:highlight_start], vals[:highlight_start], linestyle='dotted', alpha=0.2, label=f"_{cat}")
        plt.plot(x[highlight_start:], vals[highlight_start:], label=cat)

    plt.xlabel("Layer")
    plt.ylabel("Mean Abs Correlation")
    plt.title("Layerwise Bias Influence Per Category")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# 🎯 Final Heatmap Extraction (After Unlearning)

# Rehook for After-Unlearning heatmaps
hidden_states_per_layer.clear()
mlp_outputs_per_layer.clear()

hidden_hooks = []
mlp_hooks = []

for layer_idx, block in enumerate(model.model.layers):
    h1 = block.register_forward_hook(lambda module, input, output, idx=layer_idx: hook_hidden_states(module, input, output, idx))
    hidden_hooks.append(h1)
    if hasattr(block, 'mlp'):
        h2 = block.mlp.register_forward_hook(lambda module, input, output, idx=layer_idx: hook_mlp_outputs(module, input, output, idx))
        mlp_hooks.append(h2)

for idx, example in enumerate(balanced_examples):
    print(f"Saving AFTER-unlearning heatmaps for example {idx+1}/{len(balanced_examples)}...")
    save_dir = BIAS_LOCALIZATION_AFTER_DIR / f"example_{idx}"
    save_dir.mkdir(parents=True, exist_ok=True)

    question = example["question"]
    options = [example["ans0"], example["ans1"], example["ans2"]]
    context = example["context"]

    prompt = f"{context} {question}\n"
    for i, opt in enumerate(options):
        prompt += f"{chr(65+i)}. {opt}\n"
    prompt += "\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], output_attentions=True)
        attentions = outputs.attentions

    if hidden_states_per_layer:
        hs = hidden_states_per_layer[max(hidden_states_per_layer.keys())].squeeze(0)
        # hs = hs.transpose(0, 1)  # 🚨 Transpose it properly
        plot_heatmap(hs, "Hidden State Last Layer AFTER", save_dir / "hidden_last_after.png", xlabel="Hidden Size", ylabel="Tokens")

    if mlp_outputs_per_layer:
        mlp = mlp_outputs_per_layer[max(mlp_outputs_per_layer.keys())].squeeze(0)
        # mlp = mlp.transpose(0, 1)  # 🚨 Transpose it properly
        plot_heatmap(mlp, "MLP Last Layer AFTER", save_dir / "mlp_last_after.png", xlabel="Hidden Size", ylabel="Tokens")


    if attentions:
        last_attn_idx = len(attentions) - 1
        attn_map = attentions[last_attn_idx][0]
        for head_idx in range(attn_map.shape[0]):
            attn_head = attn_map[head_idx].cpu()
            plot_heatmap(attn_head, f"Attention Head {head_idx} AFTER", save_dir / f"attention_head{head_idx}_after.png")

for h in hidden_hooks + mlp_hooks:
    h.remove()

plot_layerwise_bias_curve(head_corrs_by_cat, BIAS_EVAL_DIR / "layerwise_bias_before.png")
head_corrs_by_cat_after = compute_head_correlations_by_category(model, balanced_examples, tokenizer)
plot_layerwise_bias_curve(head_corrs_by_cat_after, BIAS_MITIGATION_DIR / "layerwise_bias_after.png")


print("\n✅ Bias Mitigation and After-Unlearning Visualization Complete!")

def print_most_biased_layers(head_corrs_by_cat, top_k=3, threshold=0.05, save_path=None):
    output_lines = ["🔍 Most Biased Layers per Category:"]
    for cat in sorted(head_corrs_by_cat.keys()):
        correlations = head_corrs_by_cat[cat]
        layer_scores = {}
        for (layer, _), value in correlations.items():
            layer_scores[layer] = layer_scores.get(layer, []) + [abs(value)]

        averaged = {layer: np.mean(vals) for layer, vals in layer_scores.items() if np.mean(vals) > threshold}
        top_layers = sorted(averaged.items(), key=lambda x: x[1], reverse=True)[:top_k]

        if top_layers:
            output_lines.append(f"\n📚 {cat}:")
            for layer, score in top_layers:
                output_lines.append(f"  - Layer {layer}: Mean Abs Corr = {score:.4f}")
        else:
            output_lines.append(f"\n📚 {cat}: No strongly biased layers found.")

    for line in output_lines:
        print(line)

    if save_path is not None:
        with open(save_path, "w") as f:
            for line in output_lines:
                f.write(line + "\n")

print_most_biased_layers(head_corrs_by_cat, save_path=BIAS_EVAL_DIR / "biased_layers_before.txt")
print_most_biased_layers(head_corrs_by_cat_after, save_path=BIAS_MITIGATION_DIR / "biased_layers_after.txt")
