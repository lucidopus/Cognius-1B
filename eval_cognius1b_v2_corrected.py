import re, os, json, sys
import pandas as pd
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_secret(notebook_path, pattern):
    txt = open(notebook_path).read()
    m = re.search(pattern, txt)
    if not m:
        raise RuntimeError(f"secret pattern not found in {notebook_path}")
    return m.group(1)

HF_TOKEN = extract_secret(os.path.join(PROJECT_DIR, "Cognius.ipynb"), r"HF_TOKEN = \'(hf_[A-Za-z0-9]+)\'")
GROQ_API_KEY = extract_secret(os.path.join(PROJECT_DIR, "Critic.ipynb"), r"GROQ_API_KEY = \'(gsk_[A-Za-z0-9]+)\' # 30402")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from huggingface_hub import login

login(HF_TOKEN)

PAD_TOKEN = "<|pad|>"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
FINETUNED_REPO = "harshil30402/Cognius-mini-1B-v2"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading tokenizer/base model...", file=sys.stderr)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device, token=HF_TOKEN)

# The published adapter checkpoint includes full embed_tokens/lm_head weights
# (not just LoRA deltas), at whatever vocab size the training run actually used.
# Read that size directly from the checkpoint so our resize matches exactly,
# rather than trusting the (possibly stale) tokenizer_config.json on the repo.
from huggingface_hub import hf_hub_download
from safetensors import safe_open

adapter_weights_path = hf_hub_download(FINETUNED_REPO, "adapter_model.safetensors", token=HF_TOKEN)
with safe_open(adapter_weights_path, framework="pt") as f:
    ckpt_vocab_size = f.get_slice("base_model.model.model.embed_tokens.weight").get_shape()[0]

current_vocab_size = base_model.get_input_embeddings().weight.shape[0]
if ckpt_vocab_size != current_vocab_size:
    print(f"Resizing embeddings to match checkpoint: {current_vocab_size} -> {ckpt_vocab_size}", file=sys.stderr)
    base_model.resize_token_embeddings(ckpt_vocab_size, mean_resizing=False)

print(f"Loading LoRA adapter from {FINETUNED_REPO}...", file=sys.stderr)
finetuned_model = PeftModel.from_pretrained(base_model, FINETUNED_REPO, token=HF_TOKEN)
finetuned_model.eval()

pipe = pipeline(
    task="text-generation",
    model=finetuned_model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    return_full_text=False,
)

df = pd.read_csv(os.path.join(PROJECT_DIR, "data/preprocessed/predictions_df.csv"))
print(f"Loaded {len(df)} test examples with cached baseline (untrained) predictions.", file=sys.stderr)

finetuned_predictions = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating fine-tuned predictions"):
    out = pipe(row["prompt"])[0]["generated_text"]
    finetuned_predictions.append(out)

df["finetuned_prediction"] = finetuned_predictions
df.to_csv(os.path.join(PROJECT_DIR, "data/scores_cognius1b_v2_corrected_raw.csv"), index=False)
print("Saved raw generations to data/scores_cognius1b_v2_corrected_raw.csv", file=sys.stderr)

# free GPU/MPS memory before judging (judging is API-only, doesn't need the model)
del finetuned_model, base_model, pipe
import gc
gc.collect()
try:
    torch.mps.empty_cache()
except Exception:
    pass

from groq import Groq
client = Groq(api_key=GROQ_API_KEY)

def generate_args(question, correct_answer, model_response):
    return {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": """
You are an advanced AI model designed to critically evaluate and score model-generated answers with extreme precision. Your task is to assess responses from a Large Language model by comparing them to a ground truth answer.

Evaluation Criteria:

For each given task, evaluate the model based on the following factors:
\t1.\tAccuracy (0-10): How factually correct is the response compared to the ground truth?
\t2.\tCompleteness (0-10): Does the response fully cover the key aspects of the answer?
\t3.\tClarity & Coherence (0-10): Is the response well-structured, easy to understand, and logically coherent?
\t4.\tConciseness & Relevance (0-10): Does the response avoid unnecessary information while staying relevant?
\t5.\tDepth of Reasoning (0-10): Does the response demonstrate deep understanding, including nuanced insights or self-reflection?

"""
            },
            {
                "role": "user",
                "content": f"""

\t•\tCompare model responses to the provided ground truth answer.
\t•\tAssign scores (0-10) for each category above and calculate an overall score (average of all categories).
\t•\tHere's the data

                Question: {question} \n
                Ground Truth Answer: {correct_answer} \n
                Model's answer: {model_response}

                """
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "evaluate_models",
                    "description": "Evaluate the model response and return its scores based on the evaluation criteria.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_score": {
                                "type": "int",
                                "description": "The score (Out of 10)"
                            },
                        },
                        "required": ["model_score"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "max_completion_tokens": 4096
    }

def score(question, correct_answer, model_response, retries=3):
    for attempt in range(retries):
        try:
            comp = client.chat.completions.create(**generate_args(question, correct_answer, model_response))
            args = json.loads(comp.choices[0].message.tool_calls[0].function.arguments)
            return args["model_score"]
        except Exception as e:
            print(f"  judge error (attempt {attempt+1}/{retries}): {e}", file=sys.stderr)
    return None

standard_scores = []
finetuned_scores = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging standard (untrained) responses"):
    standard_scores.append(score(row["task"], row["answer"], row["untrained_prediction"]))

for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging fine-tuned responses"):
    finetuned_scores.append(score(row["task"], row["answer"], row["finetuned_prediction"]))

df["standard_model_score"] = standard_scores
df["finetuned_model_score"] = finetuned_scores

out_path = os.path.join(PROJECT_DIR, "data/scores_cognius1b_v2_corrected.csv")
df.to_csv(out_path, index=False)

valid = df.dropna(subset=["standard_model_score", "finetuned_model_score"])
std_mean = valid["standard_model_score"].mean()
ft_mean = valid["finetuned_model_score"].mean()
lift = ft_mean - std_mean
rel = (lift / std_mean) * 100 if std_mean else float("nan")

print(f"\n=== RESULTS (n={len(valid)}/{len(df)}) ===")
print(f"standard_model_score mean: {std_mean:.4f}")
print(f"finetuned_model_score mean: {ft_mean:.4f}")
print(f"absolute lift: {lift:.4f}")
print(f"relative improvement: {rel:.4f}%")
print(f"saved: {out_path}")
