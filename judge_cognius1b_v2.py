import os, json, sys
import pandas as pd
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_env(path):
    env = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env

env = load_env(os.path.join(PROJECT_DIR, ".env"))
GROQ_API_KEY = env["GROQ_API_KEY"]

from groq import Groq
client = Groq(api_key=GROQ_API_KEY)

def generate_args(question, correct_answer, model_response):
    return {
        "model": "openai/gpt-oss-120b",
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
                                "type": "number",
                                "description": "The score (Out of 10), the average of the 5 criteria above"
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

import re, time

def score(question, correct_answer, model_response, retries=6):
    for attempt in range(retries):
        try:
            comp = client.chat.completions.create(**generate_args(question, correct_answer, model_response))
            args = json.loads(comp.choices[0].message.tool_calls[0].function.arguments)
            return args["model_score"]
        except Exception as e:
            msg = str(e)
            wait = 5.0
            m = re.search(r"try again in ([\d.]+)s", msg)
            if m:
                wait = float(m.group(1)) + 0.5
            print(f"  judge error (attempt {attempt+1}/{retries}), waiting {wait:.1f}s: {e}", file=sys.stderr)
            time.sleep(wait)
    return None

def main():
    df = pd.read_csv(os.path.join(PROJECT_DIR, "data/scores_cognius1b_v2_corrected_raw.csv"))
    print(f"Loaded {len(df)} rows with cached generations.", file=sys.stderr)

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

if __name__ == "__main__":
    main()
