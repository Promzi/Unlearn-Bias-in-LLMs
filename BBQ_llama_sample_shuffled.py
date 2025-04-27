# from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login
# import torch, json, random
# from pathlib import Path
# import pandas as pd
# import re

# # 🔐 Login to Hugging Face (skip if already logged in)
# login(token="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")
# # ⚙️ Setup paths and config
# CACHE_DIR = "/scratch/phossai/model_cache"
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# DATA_DIR = Path("BBQ/data")
# FILES = list(DATA_DIR.glob("*.jsonl"))


# # 🧠 Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device_map="auto")
# model.to("cuda" if torch.cuda.is_available() else "cpu")
# # 📦 Sample 5 tuples from available BBQ files
# sampled_examples = []
# for file in FILES:
#     with open(file, "r") as f:
#         data = [json.loads(line) for line in f]
#         sampled = random.sample(data, min(200 - len(sampled_examples), len(data)))
#         sampled_examples.extend(sampled)
#     if len(sampled_examples) >= 200:
#         break
# print(f"🟢 Sampled {len(sampled_examples)} examples.")


# # 🧪 Run inference and collect results
# results = []
# for ex in sampled_examples:
#     context = ex["context"]
#     question = ex["question"]
#     options = [ex["ans0"], ex["ans1"], ex["ans2"]]
#     prompt = (
#         f"Context: {context}\n"
#         f"Question: {question}\n"
#         f"Options:\n"
#         f"A. {options[0]}\n"
#         f"B. {options[1]}\n"
#         f"C. {options[2]}\n\n"
#         f"Choose the best answer from the options below.\n"
#         f"Respond with only a single letter (A, B, or C).\n\n"
#         f"Answer:"
#     )
#     # 🔁 Tokenize normally without chat template
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=inputs["input_ids"],
#             max_new_tokens=5,  # shorter output
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             do_sample=False,  # deterministic
#             temperature=0.0    # avoid variation
#         )
    
#     # Get only the newly generated tokens (not including the prompt)
#     prompt_length = inputs["input_ids"].shape[1]
#     new_tokens = outputs[0][prompt_length:]
#     decoded_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
#     # Extract just the first A, B, or C from the response
#     match = re.search(r'\b([ABC])\b', decoded_answer, re.IGNORECASE)
#     response = match.group(1).upper() if match else ""
    
#     results.append({
#         "example_id": ex["example_id"],
#         "question_index": ex["question_index"],
#         "question_polarity": ex["question_polarity"],
#         "context_condition": ex["context_condition"],
#         "category": ex["category"],
#         "prompt": prompt,
#         "response": response,  # Just the letter A, B, or C
#         "label": ex["label"]
#     })
# # 💾 Save all results to CSV
# df = pd.DataFrame(results)
# df.to_csv("llama_bbq_mcq_outputs.csv", index=False)
# print("✅ Saved results to llama_bbq_mcq_outputs.csv")

###################################################PROBABILITY########################################################################

# from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login
# import torch, json, random
# from pathlib import Path
# import pandas as pd
# import re
# import numpy as np
# import torch.nn.functional as F

# # 🔐 Login to Hugging Face (skip if already logged in)
# login(token="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")

# # ⚙️ Setup paths and config
# CACHE_DIR = "/scratch/phossai/model_cache"
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# DATA_DIR = Path("BBQ/data")
# FILES = list(DATA_DIR.glob("*.jsonl"))

# # 🧠 Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device_map="auto")
# model.to("cuda" if torch.cuda.is_available() else "cpu")

# # 📦 Sample examples from available BBQ files
# sampled_examples = []
# for file in FILES:
#     with open(file, "r") as f:
#         data = [json.loads(line) for line in f]
#         sampled = random.sample(data, min(200 - len(sampled_examples), len(data)))
#         sampled_examples.extend(sampled)
#     if len(sampled_examples) >= 200:
#         break
# print(f"🟢 Sampled {len(sampled_examples)} examples.")

# # Function to calculate probabilities for answer options
# def get_option_probabilities(model, tokenizer, prompt, device="cuda"):
#     # Get token IDs for A, B, C
#     a_id = tokenizer.encode(" A", add_special_tokens=False)[0]  # Space before A is important
#     b_id = tokenizer.encode(" B", add_special_tokens=False)[0]
#     c_id = tokenizer.encode(" C", add_special_tokens=False)[0]
    
#     # Also try without space in case the model outputs them differently
#     a_id_alt = tokenizer.encode("A", add_special_tokens=False)[0]
#     b_id_alt = tokenizer.encode("B", add_special_tokens=False)[0]
#     c_id_alt = tokenizer.encode("C", add_special_tokens=False)[0]
    
#     # Tokenize prompt
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
#     # Forward pass and get logits for the next token
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits[0, -1, :]  # Get logits for next token prediction
    
#     # Get probabilities for A, B, C token IDs
#     probs = F.softmax(logits, dim=0)
    
#     # Get probabilities for both versions (with and without space)
#     prob_a = float(probs[a_id] + probs[a_id_alt])
#     prob_b = float(probs[b_id] + probs[b_id_alt])
#     prob_c = float(probs[c_id] + probs[c_id_alt])
    
#     # Normalize to sum to 1
#     total = prob_a + prob_b + prob_c
#     if total > 0:  # Avoid division by zero
#         prob_a /= total
#         prob_b /= total
#         prob_c /= total
    
#     return {"A": prob_a, "B": prob_b, "C": prob_c}

# # 🧪 Run inference and collect results
# results = []
# for ex in sampled_examples:
#     context = ex["context"]
#     question = ex["question"]
#     options = [ex["ans0"], ex["ans1"], ex["ans2"]]
#     prompt = (
#         f"Context: {context}\n"
#         f"Question: {question}\n"
#         f"Options:\n"
#         f"A. {options[0]}\n"
#         f"B. {options[1]}\n"
#         f"C. {options[2]}\n\n"
#         f"Choose the best answer from the options below.\n"
#         f"Respond with only a single letter (A, B, or C).\n\n"
#         f"Answer:"
#     )
    
#     # Get probabilities for each option
#     device = model.device
#     probs = get_option_probabilities(model, tokenizer, prompt, device)
    
#     # Generate the actual answer as before
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=inputs["input_ids"],
#             max_new_tokens=5,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             do_sample=False,
#             temperature=0.0
#         )
    
#     # Get only the newly generated tokens (not including the prompt)
#     prompt_length = inputs["input_ids"].shape[1]
#     new_tokens = outputs[0][prompt_length:]
#     decoded_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
#     # Extract just the first A, B, or C from the response
#     match = re.search(r'\b([ABC])\b', decoded_answer, re.IGNORECASE)
#     response = match.group(1).upper() if match else ""
    
#     results.append({
#         "example_id": ex["example_id"],
#         "question_index": ex["question_index"],
#         "question_polarity": ex["question_polarity"],
#         "context_condition": ex["context_condition"],
#         "category": ex["category"],
#         "prompt": prompt,
#         "response": response,  # Just the letter A, B, or C
#         "prob_A": probs["A"],
#         "prob_B": probs["B"], 
#         "prob_C": probs["C"],
#         "label": ex["label"]
#     })

# # 💾 Save all results to CSV
# df = pd.DataFrame(results)
# df.to_csv("llama_bbq_mcq_outputs_with_probs.csv", index=False)
# print("✅ Saved results with probabilities to llama_bbq_mcq_outputs_with_probs.csv")

#########################################################Shuffle, seed, Context and Prompt prefix removed#########################################

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

# 🧠 Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, device_map="auto")
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 📦 Sample examples from available BBQ files
sampled_examples = []
for file in FILES:
    with open(file, "r") as f:
        data = [json.loads(line) for line in f]
        sampled = random.sample(data, min(200 - len(sampled_examples), len(data)))
        sampled_examples.extend(sampled)
    if len(sampled_examples) >= 200:
        break
print(f"🟢 Sampled {len(sampled_examples)} examples.")

# Function to calculate probabilities for answer options
def get_option_probabilities(model, tokenizer, prompt, device="cuda"):
    # Get token IDs for A, B, C
    a_id = tokenizer.encode(" A", add_special_tokens=False)[0]  # Space before A is important
    b_id = tokenizer.encode(" B", add_special_tokens=False)[0]
    c_id = tokenizer.encode(" C", add_special_tokens=False)[0]
    
    # Also try without space in case the model outputs them differently
    a_id_alt = tokenizer.encode("A", add_special_tokens=False)[0]
    b_id_alt = tokenizer.encode("B", add_special_tokens=False)[0]
    c_id_alt = tokenizer.encode("C", add_special_tokens=False)[0]
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Forward pass and get logits for the next token
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Get logits for next token prediction
    
    # Get probabilities for A, B, C token IDs
    probs = F.softmax(logits, dim=0)
    
    # Get probabilities for both versions (with and without space)
    prob_a = float(probs[a_id] + probs[a_id_alt])
    prob_b = float(probs[b_id] + probs[b_id_alt])
    prob_c = float(probs[c_id] + probs[c_id_alt])
    
    # Normalize to sum to 1
    total = prob_a + prob_b + prob_c
    if total > 0:  # Avoid division by zero
        prob_a /= total
        prob_b /= total
        prob_c /= total
    
    return {"A": prob_a, "B": prob_b, "C": prob_c}

# 🧪 Run inference and collect results
results = []
for ex in sampled_examples:
    context = ex["context"]
    question = ex["question"]
    options = [ex["ans0"], ex["ans1"], ex["ans2"]]
    
    # Shuffle the option contents (not the letters)
    shuffled_option_indices = list(range(len(options)))
    random.shuffle(shuffled_option_indices)
    
    shuffled_options = [options[i] for i in shuffled_option_indices]
    
    # Create mapping from shuffled position to original position
    # If shuffled_option_indices is [2,0,1], this means:
    # - The option at original position 2 is now at position 0 (A)
    # - The option at original position 0 is now at position 1 (B)
    # - The option at original position 1 is now at position 2 (C)
    shuffled_to_original = {i: shuffled_option_indices[i] for i in range(len(shuffled_option_indices))}
    
    # Modified prompt format as requested
    context_question = f"{context} {question}"
    options_text = "\n".join([f"{chr(65+i)}. {shuffled_options[i]}" for i in range(len(shuffled_options))])
    
    prompt = f"{context_question}\n{options_text}\n\nAnswer:"
    
    # Get probabilities for each option
    device = model.device
    probs = get_option_probabilities(model, tokenizer, prompt, device)
    
    # Generate the actual answer as before
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.0
        )
    
    # Get only the newly generated tokens (not including the prompt)
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][prompt_length:]
    decoded_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Extract just the first A, B, or C from the response
    match = re.search(r'\b([ABC])\b', decoded_answer, re.IGNORECASE)
    response_letter = match.group(1).upper() if match else ""
    
    # Map the letter response (A/B/C) to its position (0/1/2)
    if response_letter:
        response_position = ord(response_letter) - ord('A')
        # Map that position back to the original option index
        original_option_idx = shuffled_to_original.get(response_position, -1)
        # Convert back to a letter (A/B/C)
        original_answer = chr(65 + original_option_idx) if 0 <= original_option_idx < 3 else ""
    else:
        original_option_idx = -1
        original_answer = ""
    
    # Store the mapping for clarity
    option_mapping = {chr(65+i): options[shuffled_option_indices[i]] for i in range(len(options))}
    
    results.append({
        "example_id": ex["example_id"],
        "question_index": ex["question_index"],
        "question_polarity": ex["question_polarity"],
        "context_condition": ex["context_condition"],
        "category": ex["category"],
        "prompt": prompt,
        "model_response": response_letter,  # The actual letter (A, B, C) the model output
        "original_option_picked": original_answer,  # Mapped back to original ordering
        "option_A": option_mapping.get('A', ""),
        "option_B": option_mapping.get('B', ""),
        "option_C": option_mapping.get('C', ""),
        "original_option_A": options[0],
        "original_option_B": options[1],
        "original_option_C": options[2],
        "prob_A": probs["A"],
        "prob_B": probs["B"], 
        "prob_C": probs["C"],
        "label": ex["label"]
    })

# 💾 Save all results to CSV
df = pd.DataFrame(results)
df.to_csv("llama_bbq_mcq_outputs_shuffled_fixed.csv", index=False)
print("✅ Saved results with properly shuffled options to llama_bbq_mcq_outputs_shuffled.csv")