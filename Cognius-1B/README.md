---
library_name: peft
base_model: meta-llama/Llama-3.2-1B-Instruct
license: mit
tags:
- lora
- peft
- supervised-fine-tuning
- reasoning
- self-reflection
pipeline_tag: text-generation
---

# Cognius-mini-1B-v2

A LoRA fine-tune of [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), trained via supervised fine-tuning to produce structured chain-of-thought and self-critique before answering, on a small curated reasoning dataset.

## Model Details

### Model Description

Cognius-mini-1B-v2 is Llama-3.2-1B-Instruct fine-tuned with LoRA on 1,044 curated `(question, reasoning, self-reflection, answer)` examples. The training format asks the model to think through a question, critique its own reasoning, and then answer — rather than answering directly. Evaluated against the same-size untrained base model on 42 held-out questions, judged pairwise by an LLM on a 5-criterion rubric, the fine-tuned model scored **40.8% higher on average** (8.68/10 vs. 6.16/10 for the base model).

- **Developed by:** [harshil30402](https://huggingface.co/harshil30402)
- **Model type:** Causal decoder-only LM, LoRA adapter over Llama-3.2-1B-Instruct
- **Language(s):** English
- **License:** MIT
- **Finetuned from model:** [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

### Model Sources

- **Repository:** https://huggingface.co/harshil30402/Cognius-mini-1B-v2
- **Training/eval code:** https://github.com/lucidopus/Cognius-1B

## Uses

### Direct Use

Short-form question answering where a brief, structured reasoning + self-reflection pass before the final answer is useful — e.g. as a lightweight local reasoning assistant. The model was trained on short, general-knowledge questions (science, history, concepts), not on math word problems or formal logic puzzles.

### Out-of-Scope Use

Not evaluated for factual reliability on specialized/technical domains, multi-turn dialogue, or long-context tasks. Not suitable for use where a wrong or fabricated factual answer could cause harm — outputs are not fact-checked or grounded.

## Training Details

### Training Data

- 1,044 examples, each with four fields: `task` (the question), `reasoning` (a chain-of-thought), `self_reflection` (a critique/refinement of that reasoning), and `answer` (the final answer). Topics are general open-domain knowledge questions (e.g. "How does photosynthesis work?", "What is artificial intelligence?"), not benchmark math/logic sets.
- Split into 835 train / 104 val / 105 test at the raw-example level, then reformatted into a Llama-3 chat template; after formatting, 820 train / 164 val examples were actually used for training.
- **Known limitation:** the held-out test split was not fully de-duplicated against training data — roughly 29% of test questions share task text with a training example (see [Evaluation](#evaluation)). Treat the evaluation numbers below as directionally correct rather than a clean generalization measurement.

### Training Procedure

Supervised fine-tuning with [TRL](https://github.com/huggingface/trl)'s `SFTTrainer` and a completion-only loss mask (`DataCollatorForCompletionOnlyLM`, splitting on `<|end_header_id|>` so loss is only computed on the assistant's response, not the prompt).

#### Preprocessing

- Tokenizer: Llama-3.2-1B-Instruct's tokenizer with an added `<|pad|>` pad token (right-padded); vocabulary resized to 128,264 (`pad_to_multiple_of=8` alignment).
- Chat template wraps the question, a `<think>...</think>` reasoning block, and a `<self_reflection>...</self_reflection>` block ahead of the final answer, with a system prompt instructing the model to "think and express meta cognition" before answering.
- `max_seq_length`: 512 tokens.

#### Training Hyperparameters

| Hyperparameter | Value |
|---|---|
| LoRA rank (r) | 32 |
| LoRA alpha | 16 |
| LoRA dropout | 0.005 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| LoRA bias | none |
| Learning rate | 1e-4 |
| LR scheduler | constant, 10% warmup |
| Optimizer | AdamW (torch), β=(0.9, 0.999), ε=1e-8 |
| Effective batch size | 8 (per-device 2 × gradient accumulation 4) |
| Epochs / steps | ~1 epoch (102 steps) |
| Precision | fp32 |
| Weight decay | 0.0 |
| Max grad norm | 1.0 |
| Seed | 42 |

#### Speeds, Sizes, Times

Trained locally on Apple Silicon (MPS), not a cloud GPU. Train loss fell from 1.18 to 0.86 over the 102 steps.

## Evaluation

### Testing Data, Factors & Metrics

- **Testing data:** 42 held-out questions (see the data-contamination caveat above).
- **Method:** for each question, both the untrained base model (Llama-3.2-1B-Instruct) and this fine-tuned model generated one response. An LLM judge (`openai/gpt-oss-120b`, served via Groq) scored each response 0–10 against the ground-truth answer on five criteria — Accuracy, Completeness, Clarity & Coherence, Conciseness & Relevance, and Depth of Reasoning — and the five scores were averaged into one overall score per response.
- **Note on judge model:** the original judge planned for this project (Llama-3.3-70B) was retired from Groq's API by the time this evaluation was run; `openai/gpt-oss-120b` (120B open-weight model) was used instead as the closest available substitute, with an identical rubric.

### Results

| | Mean score (0–10) |
|---|---|
| Base model (Llama-3.2-1B-Instruct, untrained) | 6.16 |
| Cognius-mini-1B-v2 (fine-tuned) | 8.68 |
| **Relative improvement** | **+40.8%** |

#### Summary

The fine-tuned model consistently produced shorter, more on-target answers matching the ground-truth style, versus the base model's more verbose, sometimes tangential, and occasionally erroneous (e.g. false-refusal) responses. Given the test-set overlap noted above, treat the magnitude as an upper-bound estimate rather than a precise generalization number.

## How to Get Started with the Model

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
adapter_name = "harshil30402/Cognius-mini-1B-v2"

tokenizer = AutoTokenizer.from_pretrained(adapter_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)

model = PeftModel.from_pretrained(base_model, adapter_name)
model.eval()

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is artificial intelligence?"}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Bias, Risks, and Limitations

- Evaluated on a small (42-example), partially train-overlapping test set — treat results as indicative, not a rigorous benchmark.
- Judged by a single LLM judge rather than multiple independent judges or human raters; no inter-rater agreement was measured.
- Trained on general-knowledge trivia-style questions; not evaluated on mathematical, logical, or multi-step reasoning benchmarks despite the "reasoning" framing.
- Outputs are not fact-checked; the model can still produce incorrect or fabricated answers.

## Technical Specifications

### Compute Infrastructure

- **Hardware:** Apple Silicon (MPS), local machine.
- **Software:** PyTorch 2.6.0, Transformers 4.50.0, PEFT 0.15.0, TRL.

## Model Card Contact

harshil30402
