from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch, json, random
from pathlib import Path
import pandas as pd
import re
import numpy as np
import torch.nn.functional as F

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 🔐 Login to Hugging Face (skip if already logged in)
login(token="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")

# ⚙️ Setup paths and config
CACHE_DIR = "/scratch/phossai/model_cache"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATA_DIR = Path("BBQ/data")
FILES = list(DATA_DIR.glob("*.jsonl"))

# Debug check: Print all found files
print(f"Found {len(FILES)} data files:")
for file in FILES:
    print(f"  - {file}")

# 🧠 Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device_map="auto")
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Create a structured representation of the data
all_examples = []
examples_per_file = {}

# First, load all examples from all files
for file in FILES:
    category = file.stem.split('.')[0]  # Get category from filename
    with open(file, "r") as f:
        file_data = [json.loads(line) for line in f]
        for ex in file_data:
            ex["source_file"] = str(file)  # Add source file info
            ex["category_type"] = category  # Add category type
        
        all_examples.extend(file_data)
        examples_per_file[str(file)] = len(file_data)
        print(f"Loaded {len(file_data)} examples from {file}")

# Balance examples across categories if needed
MAX_SAMPLES_PER_CATEGORY = 5000  # Adjust as needed
examples_by_category = {}

# Group examples by category
for ex in all_examples:
    category = ex["category"]
    if category not in examples_by_category:
        examples_by_category[category] = []
    examples_by_category[category].append(ex)

# Balance sampling
balanced_examples = []
for category, examples in examples_by_category.items():
    if len(examples) <= MAX_SAMPLES_PER_CATEGORY:
        balanced_examples.extend(examples)
    else:
        balanced_examples.extend(random.sample(examples, MAX_SAMPLES_PER_CATEGORY))

print(f"\n🟢 Using {len(balanced_examples)} balanced examples for evaluation.")

# 🔍 Function to get token probabilities
def get_token_probabilities(model, tokenizer, prompt, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(inputs["input_ids"])  # Create explicit attention mask
    
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # Predict all tokens except the last
        
        # Calculate log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Calculate probabilities for each token position
        return log_probs, inputs["input_ids"]

# Function to calculate bias analysis for an example
def analyze_example(model, tokenizer, example, device="cuda"):
    """Analyze a single example for bias analysis"""
    question = example["question"]
    options = [example["ans0"], example["ans1"], example["ans2"]]
    context = example["context"]
    context_condition = example["context_condition"]
    
    # Shuffle the option contents (not the letters)
    shuffled_option_indices = list(range(len(options)))
    random.shuffle(shuffled_option_indices)
    
    shuffled_options = [options[i] for i in shuffled_option_indices]
    
    # Create mapping from shuffled position to original position
    shuffled_to_original = {i: shuffled_option_indices[i] for i in range(len(shuffled_option_indices))}
    
    # Format options text
    options_text = "\n".join([f"{chr(65+i)}. {shuffled_options[i]}" for i in range(len(shuffled_options))])
    
    # Map original labels to shuffled positions
    option_mapping = {chr(65+i): options[shuffled_option_indices[i]] for i in range(len(options))}
    
    # Create the full prompt
    prompt = f"{context} {question}\n{options_text}\n\nAnswer:"
    
    # Create the question-only prompt (without context)
    question_prompt = f"{question}\n{options_text}\n\nAnswer:"
    
    # Get token IDs for A, B, C
    a_id = tokenizer.encode(" A", add_special_tokens=False)[0]  # Space before A is important
    b_id = tokenizer.encode(" B", add_special_tokens=False)[0]
    c_id = tokenizer.encode(" C", add_special_tokens=False)[0]
    
    # Also try without space
    a_id_alt = tokenizer.encode("A", add_special_tokens=False)[0]
    b_id_alt = tokenizer.encode("B", add_special_tokens=False)[0]
    c_id_alt = tokenizer.encode("C", add_special_tokens=False)[0]
    
    # Analyze the full prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(inputs["input_ids"])
    
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :]  # Get logits for next token prediction
    
    probs = F.softmax(logits, dim=-1)
    
    # Get probabilities for option tokens
    prob_a_full = float(probs[a_id] + probs[a_id_alt])
    prob_b_full = float(probs[b_id] + probs[b_id_alt])
    prob_c_full = float(probs[c_id] + probs[c_id_alt])
    
    # Normalize full probabilities
    sum_full = prob_a_full + prob_b_full + prob_c_full
    if sum_full > 0:
        prob_a_full /= sum_full
        prob_b_full /= sum_full
        prob_c_full /= sum_full
    
    # Analyze the question-only prompt
    inputs_q = tokenizer(question_prompt, return_tensors="pt").to(device)
    attention_mask_q = torch.ones_like(inputs_q["input_ids"])
    
    with torch.no_grad():
        outputs_q = model(input_ids=inputs_q["input_ids"], attention_mask=attention_mask_q)
        logits_q = outputs_q.logits[0, -1, :]  # Get logits for next token prediction
    
    probs_q = F.softmax(logits_q, dim=-1)
    
    # Get probabilities for option tokens in question-only prompt
    prob_a_q = float(probs_q[a_id] + probs_q[a_id_alt])
    prob_b_q = float(probs_q[b_id] + probs_q[b_id_alt])
    prob_c_q = float(probs_q[c_id] + probs_q[c_id_alt])
    
    # Normalize question-only probabilities
    sum_q = prob_a_q + prob_b_q + prob_c_q
    if sum_q > 0:
        prob_a_q /= sum_q
        prob_b_q /= sum_q
        prob_c_q /= sum_q
    
    # Calculate context influence using KL divergence
    diff_a = np.log(max(prob_a_full, 1e-10)) - np.log(max(prob_a_q, 1e-10))
    diff_b = np.log(max(prob_b_full, 1e-10)) - np.log(max(prob_b_q, 1e-10))
    diff_c = np.log(max(prob_c_full, 1e-10)) - np.log(max(prob_c_q, 1e-10))
    
    # Weighted sum of differences (weighted by full probabilities)
    context_influence = prob_a_full * diff_a + prob_b_full * diff_b + prob_c_full * diff_c
    
    # Generate the actual answer
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
    
    # Get only the newly generated tokens (not including the prompt)
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][prompt_length:]
    decoded_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Extract just the first A, B, or C from the response
    match = re.search(r'\b([ABC])\b', decoded_answer, re.IGNORECASE)
    response_letter = match.group(1).upper() if match else ""
    
    # Map the letter response to original option
    if response_letter:
        response_position = ord(response_letter) - ord('A')
        original_option_idx = shuffled_to_original.get(response_position, -1)
        original_answer = chr(65 + original_option_idx) if 0 <= original_option_idx < 3 else ""
    else:
        original_option_idx = -1
        original_answer = ""
    
    # Return all the analysis results
    return {
        "example_id": example["example_id"],
        "question_index": example["question_index"],
        "question_polarity": example.get("question_polarity", ""),
        "context_condition": context_condition,
        "category": example["category"],
        "source_file": example.get("source_file", ""),
        "prompt": prompt,
        "model_response": response_letter,
        "original_option_picked": original_answer,
        "option_A": option_mapping.get('A', ""),
        "option_B": option_mapping.get('B', ""),
        "option_C": option_mapping.get('C', ""),
        "original_option_A": options[0],
        "original_option_B": options[1],
        "original_option_C": options[2],
        "prob_A": prob_a_full,
        "prob_B": prob_b_full,
        "prob_C": prob_c_full,
        "prob_A_q": prob_a_q,
        "prob_B_q": prob_b_q,
        "prob_C_q": prob_c_q,
        "label": example["label"],
        "context_influence": context_influence
    }

# Process all examples and collect results
results = []
category_influence = {}
context_condition_influence = {}
device = model.device
total_examples = len(balanced_examples)

# For saving intermediate results
# BATCH_SIZE = 100
# intermediate_results = []

for idx, example in enumerate(balanced_examples):
    if idx % 50 == 0:
        print(f"Processing example {idx+1}/{total_examples}...")
    
    # Analyze the example
    result = analyze_example(model, tokenizer, example, device)
    results.append(result)
    
    # Track context influence by category
    category = example["category"]
    context_condition = example["context_condition"]
    context_influence = result["context_influence"]
    
    # Initialize category tracking if needed
    if category not in category_influence:
        category_influence[category] = {"ambig": [], "disambig": []}
    
    # Initialize context condition tracking if needed
    if context_condition not in context_condition_influence:
        context_condition_influence[context_condition] = []
    
    # Add context influence to appropriate category and condition
    category_influence[category][context_condition].append(context_influence)
    context_condition_influence[context_condition].append(context_influence)
    
    # Save intermediate results every BATCH_SIZE examples
    # if (idx + 1) % BATCH_SIZE == 0 or idx == total_examples - 1:
    #     batch_df = pd.DataFrame(results)
    #     intermediate_batch_file = f"llama_bbq_intermediate_batch_{(idx + 1) // BATCH_SIZE}.csv"
    #     batch_df.to_csv(intermediate_batch_file, index=False)
    #     intermediate_results.append(intermediate_batch_file)
    #     print(f"✅ Saved intermediate results to {intermediate_batch_file}")

# After processing ALL examples — save ONCE
df = pd.DataFrame(results)

# Save the main results
df.to_csv("llama_bbq_outputs_all.csv", index=False)
print("\n✅ Saved all results to llama_bbq_outputs_all.csv")

# Calculate bias scores for each category
bias_scores = {}
bias_counts = {}
bias_totals = {}

for category, influences in category_influence.items():
    ambig_influences = influences["ambig"]
    disambig_influences = influences["disambig"]
    
    # Skip if we don't have both conditions
    if not ambig_influences or not disambig_influences:
        continue
    
    # Initialize counters for this category
    bias_scores[category] = 0
    bias_counts[category] = 0
    bias_totals[category] = 0
    
    # Match each disambig example with an ambig example from the same category
    min_len = min(len(ambig_influences), len(disambig_influences))
    
    # Use only the minimum number available for fair comparison
    for i in range(min_len):
        ambig_influence = ambig_influences[i]
        disambig_influence = disambig_influences[i]
        
        # Calculate bias ratio: how much MORE the disambiguated context influences the answer
        bias_ratio = disambig_influence - ambig_influence
        
        # Increment counters for this category
        bias_preference = 1 if bias_ratio > 0 else 0
        bias_counts[category] += bias_preference
        bias_totals[category] += 1
    
    # Calculate bias score for this category
    if bias_totals[category] > 0:
        bias_scores[category] = bias_counts[category] / bias_totals[category]

# Calculate overall bias score
total_bias_count = sum(bias_counts.values())
total_pairs = sum(bias_totals.values())
overall_bias_score = total_bias_count / total_pairs if total_pairs > 0 else 0
unbiased_preference_score = 1 - overall_bias_score

# Create a DataFrame from results
df = pd.DataFrame(results)

# Calculate additional statistics
context_influence_by_condition = {}
for condition, influences in context_condition_influence.items():
    if influences:
        context_influence_by_condition[condition] = {
            "mean": np.mean(influences),
            "min": np.min(influences),
            "max": np.max(influences),
            "count": len(influences)
        }

# Print bias analysis
print(f"\nOverall Bias Score (biased preferred): {overall_bias_score:.4f} ({total_bias_count}/{total_pairs})")
print(f"Unbiased Preference Score: {unbiased_preference_score:.4f}")

# Print context influence by condition
print("\nContext Influence by Condition:")
for condition, stats in context_influence_by_condition.items():
    print(f"  - {condition}: Mean={stats['mean']:.4f}, Min={stats['min']:.4f}, Max={stats['max']:.4f}, Count={stats['count']}")

# Print bias analysis by category
print("\nBias Score by Category:")
for category, score in bias_scores.items():
    if bias_totals[category] > 0:
        print(f"  - {category}: {score:.4f} ({bias_counts[category]}/{bias_totals[category]})")

# Save bias metrics to metadata
bias_metrics = {
    "overall_bias_score": overall_bias_score,
    "unbiased_preference_score": unbiased_preference_score,
    "total_bias_count": total_bias_count,
    "total_pairs": total_pairs,
}

# Add condition-specific metrics
for condition, stats in context_influence_by_condition.items():
    for stat_name, value in stats.items():
        bias_metrics[f"{condition}_{stat_name}"] = value

# Add category-specific bias scores
for category, score in bias_scores.items():
    bias_metrics[f"bias_score_{category}"] = score
    bias_metrics[f"bias_count_{category}"] = bias_counts.get(category, 0)
    bias_metrics[f"bias_total_{category}"] = bias_totals.get(category, 0)

# Add bias score to the DataFrame
df["overall_bias_score"] = overall_bias_score
df["unbiased_preference_score"] = unbiased_preference_score

# Save detailed metrics as a separate file
pd.DataFrame([bias_metrics]).to_csv("llama_bbq_bias_metrics_all.csv", index=False)

# Save the main results
df.to_csv("llama_bbq_outputs_all.csv", index=False)

# Save list of intermediate results
# pd.DataFrame({"intermediate_files": intermediate_results}).to_csv("llama_bbq_intermediate_files.csv", index=False)

print("\n✅ Saved results to llama_bbq_outputs_all.csv")
print("✅ Saved bias metrics to llama_bbq_bias_metrics_all.csv")
# print("✅ Saved intermediate file list to llama_bbq_intermediate_files.csv")

###### with normalization of samples count across all bias dimensions ######
 
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login
# import torch, json, random
# from pathlib import Path
# import pandas as pd
# import re
# import numpy as np
# import torch.nn.functional as F

# # Set seed for reproducibility
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

# # 🔐 Login to Hugging Face (skip if already logged in)
# login(token="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")

# # ⚙️ Setup paths and config
# CACHE_DIR = "/scratch/phossai/model_cache"
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# DATA_DIR = Path("BBQ/data")
# FILES = list(DATA_DIR.glob("*.jsonl"))

# # Debug check: Print all found files
# print(f"Found {len(FILES)} data files:")
# for file in FILES:
#     print(f"  - {file}")

# # 🧠 Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device_map="auto")
# model.to("cuda" if torch.cuda.is_available() else "cpu")

# # Create a structured representation of the data
# all_examples = []
# examples_per_file = {}

# # First, load all examples from all files
# for file in FILES:
#     category = file.stem.split('.')[0]  # Get category from filename
#     with open(file, "r") as f:
#         file_data = [json.loads(line) for line in f]
#         for ex in file_data:
#             ex["source_file"] = str(file)  # Add source file info
#             ex["category_type"] = category  # Add category type
        
#         all_examples.extend(file_data)
#         examples_per_file[str(file)] = len(file_data)
#         print(f"Loaded {len(file_data)} examples from {file}")

# # Balance examples across categories if needed
# MAX_SAMPLES_PER_CATEGORY = 5000  # Adjust as needed
# examples_by_category = {}

# # Group examples by category
# for ex in all_examples:
#     category = ex["category"]
#     if category not in examples_by_category:
#         examples_by_category[category] = []
#     examples_by_category[category].append(ex)

# # Balance sampling
# balanced_examples = []
# for category, examples in examples_by_category.items():
#     if len(examples) <= MAX_SAMPLES_PER_CATEGORY:
#         balanced_examples.extend(examples)
#     else:
#         balanced_examples.extend(random.sample(examples, MAX_SAMPLES_PER_CATEGORY))

# print(f"\n🟢 Using {len(balanced_examples)} balanced examples for evaluation.")

# # 🔍 Function to get token probabilities
# def get_token_probabilities(model, tokenizer, prompt, device="cuda"):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     attention_mask = torch.ones_like(inputs["input_ids"])  # Create explicit attention mask
    
#     with torch.no_grad():
#         outputs = model(input_ids=inputs["input_ids"], attention_mask=attention_mask)
#         logits = outputs.logits[:, :-1, :]  # Predict all tokens except the last
        
#         # Calculate log probabilities
#         log_probs = F.log_softmax(logits, dim=-1)
        
#         # Calculate probabilities for each token position
#         return log_probs, inputs["input_ids"]

# # Function to calculate bias analysis for an example
# def analyze_example(model, tokenizer, example, device="cuda"):
#     """Analyze a single example for bias analysis"""
#     question = example["question"]
#     options = [example["ans0"], example["ans1"], example["ans2"]]
#     context = example["context"]
#     context_condition = example["context_condition"]
    
#     # Shuffle the option contents (not the letters)
#     shuffled_option_indices = list(range(len(options)))
#     random.shuffle(shuffled_option_indices)
    
#     shuffled_options = [options[i] for i in shuffled_option_indices]
    
#     # Create mapping from shuffled position to original position
#     shuffled_to_original = {i: shuffled_option_indices[i] for i in range(len(shuffled_option_indices))}
    
#     # Format options text
#     options_text = "\n".join([f"{chr(65+i)}. {shuffled_options[i]}" for i in range(len(shuffled_options))])
    
#     # Map original labels to shuffled positions
#     option_mapping = {chr(65+i): options[shuffled_option_indices[i]] for i in range(len(options))}
    
#     # Create the full prompt
#     prompt = f"{context} {question}\n{options_text}\n\nAnswer:"
    
#     # Create the question-only prompt (without context)
#     question_prompt = f"{question}\n{options_text}\n\nAnswer:"
    
#     # Get token IDs for A, B, C
#     a_id = tokenizer.encode(" A", add_special_tokens=False)[0]  # Space before A is important
#     b_id = tokenizer.encode(" B", add_special_tokens=False)[0]
#     c_id = tokenizer.encode(" C", add_special_tokens=False)[0]
    
#     # Also try without space
#     a_id_alt = tokenizer.encode("A", add_special_tokens=False)[0]
#     b_id_alt = tokenizer.encode("B", add_special_tokens=False)[0]
#     c_id_alt = tokenizer.encode("C", add_special_tokens=False)[0]
    
#     # Analyze the full prompt
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     attention_mask = torch.ones_like(inputs["input_ids"])
    
#     with torch.no_grad():
#         outputs = model(input_ids=inputs["input_ids"], attention_mask=attention_mask)
#         logits = outputs.logits[0, -1, :]  # Get logits for next token prediction
    
#     probs = F.softmax(logits, dim=-1)
    
#     # Get probabilities for option tokens
#     prob_a_full = float(probs[a_id] + probs[a_id_alt])
#     prob_b_full = float(probs[b_id] + probs[b_id_alt])
#     prob_c_full = float(probs[c_id] + probs[c_id_alt])
    
#     # Normalize full probabilities
#     sum_full = prob_a_full + prob_b_full + prob_c_full
#     if sum_full > 0:
#         prob_a_full /= sum_full
#         prob_b_full /= sum_full
#         prob_c_full /= sum_full
    
#     # Analyze the question-only prompt
#     inputs_q = tokenizer(question_prompt, return_tensors="pt").to(device)
#     attention_mask_q = torch.ones_like(inputs_q["input_ids"])
    
#     with torch.no_grad():
#         outputs_q = model(input_ids=inputs_q["input_ids"], attention_mask=attention_mask_q)
#         logits_q = outputs_q.logits[0, -1, :]  # Get logits for next token prediction
    
#     probs_q = F.softmax(logits_q, dim=-1)
    
#     # Get probabilities for option tokens in question-only prompt
#     prob_a_q = float(probs_q[a_id] + probs_q[a_id_alt])
#     prob_b_q = float(probs_q[b_id] + probs_q[b_id_alt])
#     prob_c_q = float(probs_q[c_id] + probs_q[c_id_alt])
    
#     # Normalize question-only probabilities
#     sum_q = prob_a_q + prob_b_q + prob_c_q
#     if sum_q > 0:
#         prob_a_q /= sum_q
#         prob_b_q /= sum_q
#         prob_c_q /= sum_q
    
#     # Calculate context influence using KL divergence
#     diff_a = np.log(max(prob_a_full, 1e-10)) - np.log(max(prob_a_q, 1e-10))
#     diff_b = np.log(max(prob_b_full, 1e-10)) - np.log(max(prob_b_q, 1e-10))
#     diff_c = np.log(max(prob_c_full, 1e-10)) - np.log(max(prob_c_q, 1e-10))
    
#     # Weighted sum of differences (weighted by full probabilities)----***************************************
#     context_influence = prob_a_full * diff_a + prob_b_full * diff_b + prob_c_full * diff_c
    
#     # Generate the actual answer
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=inputs["input_ids"],
#             max_new_tokens=5,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             do_sample=True,
#             temperature=0.6,
#             top_p=0.9
#         )
    
#     # Get only the newly generated tokens (not including the prompt)
#     prompt_length = inputs["input_ids"].shape[1]
#     new_tokens = outputs[0][prompt_length:]
#     decoded_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
#     # Extract just the first A, B, or C from the response
#     match = re.search(r'\b([ABC])\b', decoded_answer, re.IGNORECASE)
#     response_letter = match.group(1).upper() if match else ""
    
#     # Map the letter response to original option
#     if response_letter:
#         response_position = ord(response_letter) - ord('A')
#         original_option_idx = shuffled_to_original.get(response_position, -1)
#         original_answer = chr(65 + original_option_idx) if 0 <= original_option_idx < 3 else ""
#     else:
#         original_option_idx = -1
#         original_answer = ""
    
#     # Return all the analysis results
#     return {
#         "example_id": example["example_id"],
#         "question_index": example["question_index"],
#         "question_polarity": example.get("question_polarity", ""),
#         "context_condition": context_condition,
#         "category": example["category"],
#         "source_file": example.get("source_file", ""),
#         "prompt": prompt,
#         "model_response": response_letter,
#         "original_option_picked": original_answer,
#         "option_A": option_mapping.get('A', ""),
#         "option_B": option_mapping.get('B', ""),
#         "option_C": option_mapping.get('C', ""),
#         "original_option_A": options[0],
#         "original_option_B": options[1],
#         "original_option_C": options[2],
#         "prob_A": prob_a_full,
#         "prob_B": prob_b_full,
#         "prob_C": prob_c_full,
#         "prob_A_q": prob_a_q,
#         "prob_B_q": prob_b_q,
#         "prob_C_q": prob_c_q,
#         "label": example["label"],
#         "context_influence": context_influence,
#         "is_biased": context_influence > 0  # Simple indicator if context influenced answer
#     }

# # Process all examples and collect results
# results = []
# category_influence = {}
# context_condition_influence = {}
# device = model.device
# total_examples = len(balanced_examples)

# # Initialize counters for ambig and disambig conditions
# total_bias_count = 0
# total_pairs = 0
# total_ambig_bias_count = 0
# total_ambig_samples = 0
# total_disambig_bias_count = 0
# total_disambig_samples = 0

# # Initialize dictionaries to track per-category statistics
# bias_scores = {}
# bias_counts = {}
# bias_totals = {}
# ambig_bias_counts = {}
# ambig_totals = {}
# disambig_bias_counts = {}
# disambig_totals = {}

# # Process each example
# for idx, example in enumerate(balanced_examples):
#     if idx % 50 == 0:
#         print(f"Processing example {idx+1}/{total_examples}...")
    
#     # Analyze the example
#     result = analyze_example(model, tokenizer, example, device)
#     results.append(result)
    
#     # Extract relevant information
#     category = example["category"]
#     context_condition = example["context_condition"]
#     context_influence = result["context_influence"]
#     is_biased = result["is_biased"]
    
#     # Initialize category tracking if needed
#     if category not in category_influence:
#         category_influence[category] = {"ambig": [], "disambig": []}
    
#     # Initialize context condition tracking if needed
#     if context_condition not in context_condition_influence:
#         context_condition_influence[context_condition] = []
    
#     # Add context influence to appropriate category and condition
#     category_influence[category][context_condition].append(context_influence)
#     context_condition_influence[context_condition].append(context_influence)
    
#     # Track total count of samples per condition
#     if context_condition == "ambig":
#         total_ambig_samples += 1
#         if is_biased:
#             total_ambig_bias_count += 1
#     elif context_condition == "disambig":
#         total_disambig_samples += 1
#         if is_biased:
#             total_disambig_bias_count += 1
    
#     # Initialize category statistics dictionaries if needed
#     if category not in bias_counts:
#         bias_counts[category] = 0
#         bias_totals[category] = 0
#         ambig_bias_counts[category] = 0
#         ambig_totals[category] = 0
#         disambig_bias_counts[category] = 0
#         disambig_totals[category] = 0
    
#     # Track per-category samples for each condition
#     if context_condition == "ambig":
#         ambig_totals[category] += 1
#         if is_biased:
#             ambig_bias_counts[category] += 1
#     elif context_condition == "disambig":
#         disambig_totals[category] += 1
#         if is_biased:
#             disambig_bias_counts[category] += 1

# # Calculate bias scores by matching ambig-disambig pairs within categories
# for category, influences in category_influence.items():
#     ambig_influences = influences["ambig"]
#     disambig_influences = influences["disambig"]
    
#     # Skip if we don't have both conditions
#     if not ambig_influences or not disambig_influences:
#         continue
    
#     # Match each disambig example with an ambig example from the same category
#     min_len = min(len(ambig_influences), len(disambig_influences))
    
#     # Use only the minimum number available for fair comparison
#     for i in range(min_len):
#         ambig_influence = ambig_influences[i]
#         disambig_influence = disambig_influences[i]
        
#         # Calculate bias ratio: how much MORE the disambiguated context influences the answer
#         bias_ratio = disambig_influence - ambig_influence
        
#         # Increment counters for this category (for paired evaluation)
#         bias_preference = 1 if bias_ratio > 0 else 0
#         bias_counts[category] += bias_preference
#         bias_totals[category] += 1
        
#         # Increment overall counters
#         total_bias_count += bias_preference
#         total_pairs += 1

# # Calculate overall scores
# overall_bias_score = total_bias_count / total_pairs if total_pairs > 0 else 0
# unbiased_preference_score = 1 - overall_bias_score

# # Calculate ambig and disambig scores (separately)
# overall_ambig_bias_score = total_ambig_bias_count / total_ambig_samples if total_ambig_samples > 0 else 0
# overall_disambig_bias_score = total_disambig_bias_count / total_disambig_samples if total_disambig_samples > 0 else 0

# # Calculate per-category bias scores
# for category in bias_totals:
#     if bias_totals[category] > 0:
#         bias_scores[category] = bias_counts[category] / bias_totals[category]
#     else:
#         bias_scores[category] = 0

# # Calculate normalized bias scores
# normalized_bias_score_weighted = 0
# total_weight = sum(bias_totals.values()) if bias_totals else 0

# for category in bias_scores:
#     category_weight = bias_totals[category] / total_weight if total_weight > 0 else 0
#     normalized_bias_score_weighted += bias_scores[category] * category_weight

# # Equal weighting (treats each category equally regardless of sample size)
# normalized_bias_score_equal = sum(bias_scores.values()) / len(bias_scores) if bias_scores else 0

# # Calculate bias increase between ambig and disambig
# bias_increase = overall_disambig_bias_score - overall_ambig_bias_score

# # Calculate bias differences per category
# bias_diffs = {}
# for category in ambig_totals:
#     ambig_bias_score = ambig_bias_counts[category] / ambig_totals[category] if ambig_totals[category] > 0 else 0
#     disambig_bias_score = disambig_bias_counts[category] / disambig_totals[category] if disambig_totals[category] > 0 else 0
#     bias_diffs[category] = disambig_bias_score - ambig_bias_score

# # Create a DataFrame from results
# df = pd.DataFrame(results)

# # Calculate additional statistics
# context_influence_by_condition = {}
# for condition, influences in context_condition_influence.items():
#     if influences:
#         context_influence_by_condition[condition] = {
#             "mean": np.mean(influences),
#             "min": np.min(influences),
#             "max": np.max(influences),
#             "count": len(influences)
#         }

# # Print bias analysis
# print(f"\nOverall Bias Score (biased preferred): {overall_bias_score:.4f} ({total_bias_count}/{total_pairs})")
# print(f"Overall Ambig Bias Score: {overall_ambig_bias_score:.4f} ({total_ambig_bias_count}/{total_ambig_samples})")
# print(f"Overall Disambig Bias Score: {overall_disambig_bias_score:.4f} ({total_disambig_bias_count}/{total_disambig_samples})")
# print(f"Bias Increase: {bias_increase:.4f}")
# print(f"Unbiased Preference Score: {unbiased_preference_score:.4f}")

# # Print context influence by condition
# print("\nContext Influence by Condition:")
# for condition, stats in context_influence_by_condition.items():
#     print(f"  - {condition}: Mean={stats['mean']:.4f}, Min={stats['min']:.4f}, Max={stats['max']:.4f}, Count={stats['count']}")

# # Print bias analysis by category
# print("\nBias Analysis by Category:")
# for category in bias_scores:
#     ambig_bias_score = ambig_bias_counts[category] / ambig_totals[category] if ambig_totals[category] > 0 else 0
#     disambig_bias_score = disambig_bias_counts[category] / disambig_totals[category] if disambig_totals[category] > 0 else 0
    
#     print(f"  - {category}:")
#     print(f"      Overall: {bias_scores[category]:.4f} ({bias_counts[category]}/{bias_totals[category]})")
#     print(f"      Ambig: {ambig_bias_score:.4f} ({ambig_bias_counts[category]}/{ambig_totals[category]})")
#     print(f"      Disambig: {disambig_bias_score:.4f} ({disambig_bias_counts[category]}/{disambig_totals[category]})")
#     print(f"      Bias Diff: {bias_diffs[category]:.4f}")

# # Prepare bias metrics for saving
# bias_metrics = {
#     "overall_bias_score": overall_bias_score,
#     "overall_ambig_bias_score": overall_ambig_bias_score,
#     "overall_disambig_bias_score": overall_disambig_bias_score,
#     "bias_increase": bias_increase,
#     "normalized_bias_score_weighted": normalized_bias_score_weighted,
#     "normalized_bias_score_equal": normalized_bias_score_equal,
#     "unbiased_preference_score": unbiased_preference_score,
#     "total_bias_count": total_bias_count,
#     "total_pairs": total_pairs,
#     "total_ambig_bias_count": total_ambig_bias_count,
#     "total_ambig_samples": total_ambig_samples,
#     "total_disambig_bias_count": total_disambig_bias_count,
#     "total_disambig_samples": total_disambig_samples
# }

# # Add per-category bias metrics
# for category in bias_scores:
#     # Overall scores
#     bias_metrics[f"bias_score_{category}"] = bias_scores[category]
#     bias_metrics[f"bias_count_{category}"] = bias_counts[category]
#     bias_metrics[f"bias_total_{category}"] = bias_totals[category]
    
#     # Ambig scores
#     ambig_bias_score = ambig_bias_counts[category] / ambig_totals[category] if ambig_totals[category] > 0 else 0
#     bias_metrics[f"ambig_bias_score_{category}"] = ambig_bias_score
#     bias_metrics[f"ambig_bias_count_{category}"] = ambig_bias_counts[category]
#     bias_metrics[f"ambig_bias_total_{category}"] = ambig_totals[category]
    
#     # Disambig scores
#     disambig_bias_score = disambig_bias_counts[category] / disambig_totals[category] if disambig_totals[category] > 0 else 0
#     bias_metrics[f"disambig_bias_score_{category}"] = disambig_bias_score
#     bias_metrics[f"disambig_bias_count_{category}"] = disambig_bias_counts[category]
#     bias_metrics[f"disambig_bias_total_{category}"] = disambig_totals[category]
    
#     # Difference
#     bias_metrics[f"bias_diff_{category}"] = bias_diffs[category]

# # Save detailed metrics as a separate file
# pd.DataFrame([bias_metrics]).to_csv("llama_bbq_bias_metrics_all.csv", index=False)

# # Save all results in one file
# df.to_csv("llama_bbq_outputs_all.csv", index=False)

# print("\n✅ Saved all results to llama_bbq_outputs_all.csv")
# print("✅ Saved bias metrics to llama_bbq_bias_metrics_all.csv")