# # ======================
# # 📦 Imports
# # ======================
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login
# import torch, json, random, os
# from pathlib import Path
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import re
# import numpy as np
# import torch.nn.functional as F

# # ======================
# # ⚙️ Setup
# # ======================
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

# login(token="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")

# CACHE_DIR = "/scratch/phossai/model_cache"
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# DATA_DIR = Path("BBQ/data")
# FILES = list(DATA_DIR.glob("*.jsonl"))

# SAVE_DIR = Path("layerwise_tracing_outputs")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)

# print(f"Found {len(FILES)} data files:")
# for file in FILES:
#     print(f"  - {file}")

# # ======================
# # 🧠 Load Model and Tokenizer
# # ======================
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device_map="auto")
# model.eval()

# device = model.device

# # ======================
# # 📚 Helper Functions
# # ======================

# def get_clean_and_corrupt_prompts(example):
#     """Return (clean_prompt, corrupt_prompt) given an example."""
#     question = example["question"]
#     options = [example["ans0"], example["ans1"], example["ans2"]]
#     context = example["context"]

#     # Shuffle options
#     shuffled_option_indices = list(range(len(options)))
#     random.shuffle(shuffled_option_indices)
#     shuffled_options = [options[i] for i in shuffled_option_indices]

#     options_text = "\n".join([f"{chr(65+i)}. {shuffled_options[i]}" for i in range(len(shuffled_options))])

#     clean_prompt = f"{context} {question}\n{options_text}\n\nAnswer:"
#     corrupt_context = "[MASK]"  # Mask sensitive context
#     corrupt_prompt = f"{corrupt_context} {question}\n{options_text}\n\nAnswer:"

#     return clean_prompt, corrupt_prompt

# def capture_hidden_states(prompt, model, tokenizer):
#     """Forward pass through model capturing hidden states."""
#     hidden_states_per_layer = {}

#     def hook_hidden_states(module, input, output, layer_idx):
#         if isinstance(output, tuple):
#             output = output[0]
#         hidden_states_per_layer[layer_idx] = output.detach()

#     hooks = []
#     for layer_idx, block in enumerate(model.model.layers):
#         h = block.register_forward_hook(lambda module, input, output, idx=layer_idx: hook_hidden_states(module, input, output, idx))
#         hooks.append(h)

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model(**inputs)

#     for h in hooks:
#         h.remove()

#     return hidden_states_per_layer

# def predict_from_hidden_state(hidden_state_last, model):
#     """Predict the next token from a given last hidden state."""
#     lm_head = model.lm_head
#     logits = lm_head(hidden_state_last)  # (batch_size, seq_len, vocab_size)
#     logits = logits[:, -1, :]  # Take last token
#     probs = F.softmax(logits, dim=-1)
#     return probs

# def measure_context_influence(probs, tokenizer):
#     """Compute context influence from probabilities over A, B, C choices."""
#     a_id = tokenizer.encode(" A", add_special_tokens=False)[0]
#     b_id = tokenizer.encode(" B", add_special_tokens=False)[0]
#     c_id = tokenizer.encode(" C", add_special_tokens=False)[0]
#     a_id_alt = tokenizer.encode("A", add_special_tokens=False)[0]
#     b_id_alt = tokenizer.encode("B", add_special_tokens=False)[0]
#     c_id_alt = tokenizer.encode("C", add_special_tokens=False)[0]

#     prob_a = float(probs[0, a_id] + probs[0, a_id_alt])
#     prob_b = float(probs[0, b_id] + probs[0, b_id_alt])
#     prob_c = float(probs[0, c_id] + probs[0, c_id_alt])

#     sum_probs = prob_a + prob_b + prob_c
#     if sum_probs > 0:
#         prob_a /= sum_probs
#         prob_b /= sum_probs
#         prob_c /= sum_probs

#     # Just return entropy (uncertainty reduction measure)
#     context_influence = -(prob_a*np.log(prob_a+1e-10) + prob_b*np.log(prob_b+1e-10) + prob_c*np.log(prob_c+1e-10))
#     return context_influence

# def layerwise_restore_and_trace(clean_hiddens, corrupt_hiddens, model, tokenizer):
#     """Layerwise restoration: for each layer, restore clean hidden and measure context influence."""
#     layer_context_influences = {}

#     num_layers = len(clean_hiddens)

#     for layer_idx in range(num_layers):
#         restored_hiddens = []
#         for l in range(num_layers):
#             if l == layer_idx:
#                 restored_hiddens.append(clean_hiddens[l])
#             else:
#                 restored_hiddens.append(corrupt_hiddens[l])

#         # Take final hidden
#         final_hidden = restored_hiddens[-1]

#         probs = predict_from_hidden_state(final_hidden, model)
#         context_influence = measure_context_influence(probs, tokenizer)

#         layer_context_influences[layer_idx] = context_influence

#     return layer_context_influences

# # ======================
# # 🚀 Main Tracing Loop
# # ======================
# # Load BBQ examples
# all_examples = []
# for file in FILES:
#     category = file.stem.split('.')[0]
#     with open(file, "r") as f:
#         file_data = [json.loads(line) for line in f]
#         for ex in file_data:
#             ex["category_type"] = category
#         all_examples.extend(file_data)

# # Sample 3 examples per category (adjust if you want)
# examples_by_category = {}
# for ex in all_examples:
#     cat = ex["category"]
#     examples_by_category.setdefault(cat, []).append(ex)

# balanced_examples = []
# for cat, ex_list in examples_by_category.items():
#     if len(ex_list) >= 3:
#         balanced_examples.extend(random.sample(ex_list, 3))
#     else:
#         balanced_examples.extend(ex_list)

# print(f"\n🟢 Using {len(balanced_examples)} examples for layerwise tracing.")

# # Store results
# tracing_results = []

# # Main tracing
# for idx, example in enumerate(balanced_examples):
#     print(f"\n🧠 Processing Example {idx+1}/{len(balanced_examples)}: Category = {example['category']}")

#     clean_prompt, corrupt_prompt = get_clean_and_corrupt_prompts(example)

#     # Capture hidden states
#     hidden_clean = capture_hidden_states(clean_prompt, model, tokenizer)
#     hidden_corrupt = capture_hidden_states(corrupt_prompt, model, tokenizer)

#     # Predict directly
#     final_clean = hidden_clean[max(hidden_clean.keys())]
#     final_corrupt = hidden_corrupt[max(hidden_corrupt.keys())]

#     prob_clean = predict_from_hidden_state(final_clean, model)
#     prob_corrupt = predict_from_hidden_state(final_corrupt, model)

#     context_influence_clean = measure_context_influence(prob_clean, tokenizer)
#     context_influence_corrupt = measure_context_influence(prob_corrupt, tokenizer)

#     # Layer-by-layer restore
#     layer_influences = layerwise_restore_and_trace(hidden_clean, hidden_corrupt, model, tokenizer)

#     for layer_idx, restored_context_influence in layer_influences.items():
#         tracing_results.append({
#             "example_id": example["example_id"],
#             "category": example["category"],
#             "layer_idx": layer_idx,
#             "context_influence_clean": context_influence_clean,
#             "context_influence_corrupt": context_influence_corrupt,
#             "context_influence_restored": restored_context_influence,
#             "delta_bias_reduction": restored_context_influence - context_influence_corrupt,
#         })

# # ======================
# # 📊 Save Tracing Results
# # ======================
# tracing_df = pd.DataFrame(tracing_results)
# tracing_df.to_csv(SAVE_DIR / "layerwise_bias_tracing.csv", index=False)
# print("\n✅ Saved layerwise_bias_tracing.csv!")

# # ======================
# # 📊 Plot Tracing Results
# # ======================
# # Plot Layer vs Delta Bias for each Category
# categories = tracing_df["category"].unique()

# for cat in categories:
#     cat_df = tracing_df[tracing_df["category"] == cat]

#     plt.figure(figsize=(10,6))
#     sns.lineplot(data=cat_df, x="layer_idx", y="delta_bias_reduction", estimator='mean', ci="sd")
#     plt.title(f"Layer-by-Layer Bias Reduction - {cat}")
#     plt.xlabel("Layer Index")
#     plt.ylabel("Delta Bias Reduction (Restored vs Corrupt)")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(SAVE_DIR / f"layerwise_bias_reduction_{cat}.png")
#     plt.show()

# print("\n✅ Finished Layerwise Causal Tracing Analysis!")


# # Load the layerwise tracing results
# df = pd.read_csv(SAVE_DIR / "layerwise_bias_tracing.csv")

# plt.figure(figsize=(12,8))
# sns.lineplot(data=df, x="layer_idx", y="delta_bias_reduction", hue="category", estimator='mean', errorbar='sd', palette="tab10")
# plt.title("Layerwise Bias Reduction Across All Categories")
# plt.xlabel("Layer Index")
# plt.ylabel("Delta Bias Reduction (Restored vs Corrupt)")
# plt.grid(True)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig(SAVE_DIR / "layerwise_bias_reduction_all_categories.png")
# plt.show()

# print("\n✅ Saved combined plot to layerwise_bias_reduction_all_categories.png!")


# ===========================================================================================

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

# ================= Setup ==================
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

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device_map="auto")
model.eval()
device = model.device

# Load BBQ examples
all_examples = []
for file in FILES:
    category = file.stem.split('.')[0]
    with open(file, "r") as f:
        file_data = [json.loads(line) for line in f]
        for ex in file_data:
            ex["category_type"] = category
        all_examples.extend(file_data)
    print(f"Loaded {len(file_data)} examples from {file}")

examples_by_category = {}
for ex in all_examples:
    category = ex["category"]
    examples_by_category.setdefault(category, []).append(ex)

# ================= Hooks ==================
hidden_states_clean = {}
hidden_states_corrupt = {}

def capture_hook_clean(layer_idx):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        hidden_states_clean[layer_idx] = output.detach()
    return hook

def capture_hook_corrupt(layer_idx):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        hidden_states_corrupt[layer_idx] = output.detach()
    return hook

# ================= Analysis ==================

def analyze_example(example, layer_count):
    question = example["question"]
    options = [example["ans0"], example["ans1"], example["ans2"]]
    context = example["context"]

    shuffled_idx = list(range(3))
    random.shuffle(shuffled_idx)
    shuffled_options = [options[i] for i in shuffled_idx]
    options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(shuffled_options)])

    prompt = f"{context} {question}\n{options_text}\n\nAnswer:"
    question_prompt = f"{question}\n{options_text}\n\nAnswer:"

    inputs_clean = tokenizer(prompt, return_tensors="pt").to(device)
    inputs_corrupt = tokenizer(question_prompt, return_tensors="pt").to(device)

    clean_hooks = []
    for idx, block in enumerate(model.model.layers):
        clean_hooks.append(block.register_forward_hook(capture_hook_clean(idx)))

    with torch.no_grad():
        outputs_clean = model(**inputs_clean)

    for h in clean_hooks:
        h.remove()

    corrupt_hooks = []
    for idx, block in enumerate(model.model.layers):
        corrupt_hooks.append(block.register_forward_hook(capture_hook_corrupt(idx)))

    with torch.no_grad():
        outputs_corrupt = model(**inputs_corrupt)

    for h in corrupt_hooks:
        h.remove()

    logits_clean = outputs_clean.logits[0, -1, :]
    logits_corrupt = outputs_corrupt.logits[0, -1, :]

    probs_clean = F.softmax(logits_clean, dim=-1)
    probs_corrupt = F.softmax(logits_corrupt, dim=-1)

    a_id = tokenizer.encode(" A", add_special_tokens=False)[0]
    b_id = tokenizer.encode(" B", add_special_tokens=False)[0]
    c_id = tokenizer.encode(" C", add_special_tokens=False)[0]

    context_influence_corrupt = float(probs_clean[a_id] - probs_corrupt[a_id]) + \
                                float(probs_clean[b_id] - probs_corrupt[b_id]) + \
                                float(probs_clean[c_id] - probs_corrupt[c_id])

    layerwise_results = []
    
    for restore_idx in range(layer_count):
        with torch.no_grad():
            # Create a copy of the model inputs
            inputs = {k: v.clone() for k, v in inputs_corrupt.items()}
            
            # Run through the model's normal forward pipeline
            # But intercept and swap at the specified layer
            outputs = model.model.embed_tokens(inputs["input_ids"])
            
            # Run through layers
            for idx in range(layer_count):
                # Use the full model pipeline with proper attention mask handling
                if idx == restore_idx:
                    # Replace with clean hidden states from the same layer
                    outputs = hidden_states_clean[idx]
                else:
                    # Process normally through the layer
                    layer_outputs = model.model.layers[idx](
                        hidden_states=outputs,
                        attention_mask=inputs.get("attention_mask", None),
                        position_ids=None,  # Let the model handle this internally
                    )
                    outputs = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            
            # Final norm and projection 
            outputs = model.model.norm(outputs)
            logits = model.lm_head(outputs)
            
            # Get the last token's logits
            logits_restored = logits[0, -1, :]
            probs_restored = F.softmax(logits_restored, dim=-1)
            
            # Calculate context influence
            context_influence_restored = float(probs_clean[a_id] - probs_restored[a_id]) + \
                                         float(probs_clean[b_id] - probs_restored[b_id]) + \
                                         float(probs_clean[c_id] - probs_restored[c_id])
            
            delta_bias = context_influence_restored - context_influence_corrupt
            
            layerwise_results.append({
                "example_id": example["example_id"],
                "category": example["category"],
                "layer_idx": restore_idx,
                "delta_bias_reduction": delta_bias,
            })
                
    return layerwise_results

# ================= Main Runner ==================

def run_layerwise(SAMPLES_PER_CATEGORY, output_dir_name):
    SAVE_DIR = Path(output_dir_name)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    balanced_examples = []
    for category, examples in examples_by_category.items():
        if len(examples) <= SAMPLES_PER_CATEGORY:
            balanced_examples.extend(examples)
        else:
            balanced_examples.extend(random.sample(examples, SAMPLES_PER_CATEGORY))

    print(f"\n🟢 Using {len(balanced_examples)} examples.")

    layer_count = len(model.model.layers)
    all_results = []

    for idx, example in enumerate(balanced_examples):
        print(f"Processing Example {idx+1}/{len(balanced_examples)}")
        results = analyze_example(example, layer_count)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv(SAVE_DIR / "layerwise_bias_tracing.csv", index=False)

    plt.figure(figsize=(12,8))
    sns.lineplot(data=df, x="layer_idx", y="delta_bias_reduction", hue="category", estimator='mean', errorbar='sd')
    plt.title(f"Layerwise Bias Reduction ({SAMPLES_PER_CATEGORY} samples/category)")
    plt.xlabel("Layer Index")
    plt.ylabel("Delta Bias Reduction")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "layerwise_bias_reduction.png")
    plt.show()

    print(f"✅ Saved results to {SAVE_DIR}.")

# ================= Execute ==================

sample_sizes = [20, 50, 100]
for size in sample_sizes:
    run_layerwise(SAMPLES_PER_CATEGORY=size, output_dir_name=f"bias_localization_outputs_{size}_samples")

# 📂 Adjust this path to where your layerwise_bias_tracing.csv is located
SAVE_DIR = Path("bias_localization_outputs_500_samples")
TRACE_CSV = SAVE_DIR / "layerwise_bias_tracing.csv"

# ✅ Load the layerwise results
print(f"Loading from: {TRACE_CSV}")
df = pd.read_csv(TRACE_CSV)

# ==========================
# 🧠 Find Top Bias Reduction Layers per Example
# ==========================
# For each (example_id), find the layer where delta_bias_reduction was maximum
best_layers = df.groupby('example_id').apply(lambda x: x.loc[x['delta_bias_reduction'].idxmax()])
best_layers.reset_index(drop=True, inplace=True)

# ==========================
# 📊 Plot: Histogram of Top Bias Layers for All Examples (Overall)
# ==========================
plt.figure(figsize=(10,6))
sns.histplot(best_layers['layer_idx'], bins=len(df['layer_idx'].unique()), kde=False)
plt.title("Top Layer of Maximum Bias Reduction (All Categories)")
plt.xlabel("Layer Index")
plt.ylabel("# Examples")
plt.grid(True)
plt.tight_layout()
plt.savefig(SAVE_DIR / "top_bias_layer_histogram_all.png")
plt.show()

print("✅ Saved Top Layer Histogram for All Categories!")

# ==========================
# 📊 Plot: Top Bias Layers by Category (Separate Plots)
# ==========================

categories = best_layers['category'].unique()

for category in categories:
    plt.figure(figsize=(10,6))
    cat_layers = best_layers[best_layers['category'] == category]
    sns.histplot(cat_layers['layer_idx'], bins=len(df['layer_idx'].unique()), kde=False)
    plt.title(f"Top Layer of Maximum Bias Reduction ({category})")
    plt.xlabel("Layer Index")
    plt.ylabel("# Examples")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"top_bias_layer_histogram_{category}.png")
    plt.close()

print("✅ Saved Top Layer Histograms per Category!")

# ==========================
# 🏆 Optional: Table - Category-wise Bias Layer Averages
# ==========================

avg_layer_per_category = best_layers.groupby('category')['layer_idx'].mean().sort_values()

print("\n🏆 Average Top Bias Layer per Category:")
print(avg_layer_per_category)

avg_layer_per_category.to_csv(SAVE_DIR / "avg_top_bias_layer_per_category.csv")

print("✅ Saved Average Top Bias Layers per Category!")

