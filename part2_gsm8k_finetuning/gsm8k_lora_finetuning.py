"""
==============================================================================
   PART 2: GSM8K Fine-Tuning with LoRA on LLaMA 3.2 1B
   Vexoo Labs AI Engineer Assignment — Pranav Sharma
==============================================================================

This script is designed to run in Google Colab with a T4 GPU runtime.
Copy each section into a separate Colab cell, or upload this file and
run it directly.

Model:     meta-llama/Llama-3.2-1B (or fallback: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
Dataset:   GSM8K (3000 train / 1000 test)
Method:    LoRA-based Supervised Fine-Tuning (SFT)
Framework: HuggingFace transformers + peft + trl
"""

# ════════════════════════════════════════════════════════════════
# CELL 1: Environment Setup
# ════════════════════════════════════════════════════════════════

# !pip install -q torch transformers peft datasets accelerate bitsandbytes trl scipy

import os
import re
import json
import time
import logging
from datetime import datetime

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

# Verify GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    logger.warning("No GPU detected! Training will be very slow on CPU.")

print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")


# ════════════════════════════════════════════════════════════════
# CELL 2: Configuration
# ════════════════════════════════════════════════════════════════

# --- Model Configuration ---
# Primary: LLaMA 3.2 1B (requires HuggingFace access token)
# Fallback: TinyLlama 1.1B (openly available, similar architecture)
MODEL_NAME = "meta-llama/Llama-3.2-1B"
FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- Dataset Configuration ---
TRAIN_SAMPLES = 3000
TEST_SAMPLES = 1000
RANDOM_SEED = 42
MAX_SEQ_LENGTH = 512

# --- LoRA Configuration ---
LORA_R = 8              # Low-rank dimension (lower = fewer params, faster)
LORA_ALPHA = 16          # Scaling factor (typically 2x rank)
LORA_DROPOUT = 0.05      # Regularization dropout
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # Attention layers to adapt

# --- Training Configuration ---
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 8   # Effective batch = 4 × 8 = 32
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
SAVE_STRATEGY = "epoch"

# --- Output Paths ---
OUTPUT_DIR = "./gsm8k_lora_output"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
LOG_DIR = f"{OUTPUT_DIR}/logs"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Configuration loaded successfully.")
print(f"  Model: {MODEL_NAME}")
print(f"  Train/Test: {TRAIN_SAMPLES}/{TEST_SAMPLES}")
print(f"  LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")


# ════════════════════════════════════════════════════════════════
# CELL 3: Dataset Loading & Preparation
# ════════════════════════════════════════════════════════════════

def load_gsm8k_data(train_n=TRAIN_SAMPLES, test_n=TEST_SAMPLES, seed=RANDOM_SEED):
    """
    Load GSM8K dataset from HuggingFace and prepare train/test splits.
    
    GSM8K contains ~7.5k grade-school math problems with step-by-step
    chain-of-thought solutions ending in "#### <answer>".
    
    Args:
        train_n: Number of training samples
        test_n: Number of test samples
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    logger.info("Loading GSM8K dataset from HuggingFace...")
    dataset = load_dataset("openai/gsm8k", "main")
    
    # Shuffle with fixed seed for reproducibility
    full_train = dataset["train"].shuffle(seed=seed)
    
    # Select subsets
    train_data = full_train.select(range(min(train_n, len(full_train))))
    test_data = dataset["test"].select(range(min(test_n, len(dataset["test"]))))
    
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    # Preview a sample
    sample = train_data[0]
    print(f"\n--- Sample ---")
    print(f"Question: {sample['question'][:100]}...")
    print(f"Answer: {sample['answer'][:100]}...")
    
    return train_data, test_data


train_dataset, test_dataset = load_gsm8k_data()


# ════════════════════════════════════════════════════════════════
# CELL 4: Prompt Formatting
# ════════════════════════════════════════════════════════════════

def format_gsm8k_prompt(example):
    """
    Format a GSM8K example into a prompt-completion pair.
    
    Format:
        Question: {question}
        Answer: Let's solve this step by step.
        {chain_of_thought}
        #### {final_answer}
    
    The chain-of-thought is preserved from the original dataset
    to enable the model to learn step-by-step reasoning.
    
    Args:
        example: Dictionary with 'question' and 'answer' keys
    
    Returns:
        Dictionary with 'text' key containing the formatted prompt
    """
    question = example["question"].strip()
    answer = example["answer"].strip()
    
    formatted = (
        f"Question: {question}\n"
        f"Answer: Let's solve this step by step.\n"
        f"{answer}"
    )
    
    return {"text": formatted}


# Apply formatting to both splits
train_formatted = train_dataset.map(format_gsm8k_prompt, remove_columns=train_dataset.column_names)
test_formatted = test_dataset.map(format_gsm8k_prompt, remove_columns=test_dataset.column_names)

print(f"\nFormatted sample:")
print(train_formatted[0]["text"][:300])


# ════════════════════════════════════════════════════════════════
# CELL 5: Model & Tokenizer Loading
# ════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_name=MODEL_NAME, fallback=FALLBACK_MODEL):
    """
    Load the base model and tokenizer.
    
    Attempts to load the primary model (LLaMA 3.2 1B). If access
    is denied or the model is unavailable, falls back to TinyLlama.
    
    Uses 4-bit quantization (QLoRA-style) to reduce VRAM usage:
        - BFloat16 compute dtype for stability
        - NF4 quantization type for better accuracy
        - Double quantization for additional memory savings
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Try primary model, fall back if needed
    try:
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info(f"Successfully loaded {model_name}")
    except Exception as e:
        logger.warning(f"Could not load {model_name}: {e}")
        logger.info(f"Falling back to {fallback}")
        tokenizer = AutoTokenizer.from_pretrained(fallback, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            fallback,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info(f"Successfully loaded fallback model {fallback}")
    
    # Set padding token (LLaMA doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare model for quantized training
    model = prepare_model_for_kbit_training(model)
    
    # Print model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel loaded: {total_params / 1e6:.1f}M total parameters")
    print(f"Trainable (before LoRA): {trainable / 1e6:.1f}M parameters")
    
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()


# ════════════════════════════════════════════════════════════════
# CELL 6: LoRA Configuration & Application
# ════════════════════════════════════════════════════════════════

def apply_lora(model):
    """
    Apply LoRA (Low-Rank Adaptation) to the base model.
    
    LoRA injects trainable rank decomposition matrices into
    attention layers, enabling fine-tuning with <2% of total params.
    
    Configuration:
        - r=8: Low-rank dimension (balance between capacity and efficiency)
        - alpha=16: Scaling factor (2× rank is a common heuristic)
        - Targets: q_proj, v_proj (query and value projection in attention)
        - Dropout: 0.05 for light regularization
    
    Args:
        model: Base language model
    
    Returns:
        PEFT model with LoRA adapters attached
    """
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    peft_model = get_peft_model(model, lora_config)
    
    # Print trainable parameter stats
    peft_model.print_trainable_parameters()
    
    return peft_model


peft_model = apply_lora(model)


# ════════════════════════════════════════════════════════════════
# CELL 7: Tokenization
# ════════════════════════════════════════════════════════════════

def tokenize_function(examples):
    """
    Tokenize formatted text for causal language modeling.
    
    Uses the LLaMA tokenizer with:
        - max_length=512 (covers most GSM8K problems)
        - Truncation for safety
        - Padding to max_length for batching efficiency
    
    Labels are set equal to input_ids for causal LM training
    (next-token prediction).
    """
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


# Tokenize datasets
logger.info("Tokenizing datasets...")
train_tokenized = train_formatted.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing train"
)

test_tokenized = test_formatted.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing test"
)

# Set format for PyTorch
train_tokenized.set_format("torch")
test_tokenized.set_format("torch")

print(f"\nTokenized train: {len(train_tokenized)} samples")
print(f"Tokenized test: {len(test_tokenized)} samples")
print(f"Sequence length: {MAX_SEQ_LENGTH}")

# Count truncated samples for logging
truncated = sum(
    1 for i in range(len(train_tokenized))
    if train_tokenized[i]["attention_mask"][-1] == 1
)
print(f"Samples hitting max_length: {truncated}/{len(train_tokenized)}")


# ════════════════════════════════════════════════════════════════
# CELL 8: Training Setup & Execution
# ════════════════════════════════════════════════════════════════

def train_model(model, train_data, eval_data, tokenizer):
    """
    Configure and execute the training loop.
    
    Training Configuration:
        - 3 epochs over 3000 samples
        - Batch size 4 × gradient accumulation 8 = effective batch 32
        - AdamW optimizer with cosine LR schedule
        - Mixed precision (fp16) for memory efficiency
        - Loss logged every 50 steps
        - Checkpoints saved at each epoch
    
    Args:
        model: PEFT model with LoRA adapters
        train_data: Tokenized training dataset
        eval_data: Tokenized evaluation dataset
        tokenizer: Tokenizer instance
    
    Returns:
        Trainer instance (for later evaluation)
    """
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        fp16=True,                   # Mixed precision for speed + memory
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy=SAVE_STRATEGY,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",           # Disable wandb/tensorboard for simplicity
        optim="adamw_torch",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    
    # Data collator for causal LM (handles padding/masking)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("Starting training...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed / 60:.1f} minutes")
    
    # Log training metrics
    metrics = train_result.metrics
    metrics["training_time_minutes"] = round(elapsed / 60, 2)
    
    print(f"\n--- Training Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Save metrics to file
    with open(f"{LOG_DIR}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return trainer


trainer = train_model(peft_model, train_tokenized, test_tokenized, tokenizer)


# ════════════════════════════════════════════════════════════════
# CELL 9: Evaluation — Exact Match Accuracy
# ════════════════════════════════════════════════════════════════

def extract_numeric_answer(text):
    """
    Extract the final numeric answer from GSM8K format.
    
    GSM8K answers end with "#### <number>". This function
    extracts that number for exact match comparison.
    
    Args:
        text: Generated or ground truth text
    
    Returns:
        Extracted number as string, or None if not found
    """
    # Look for the #### pattern (GSM8K standard format)
    match = re.search(r'####\s*(\-?\d[\d,]*\.?\d*)', text)
    if match:
        return match.group(1).replace(",", "")
    
    # Fallback: try to find the last number in the text
    numbers = re.findall(r'\-?\d[\d,]*\.?\d*', text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return None


def evaluate_model(trainer, test_data, test_formatted, num_samples=100):
    """
    Evaluate the fine-tuned model on GSM8K test samples.
    
    Process:
        1. Generate completions for each test question
        2. Extract predicted numeric answer via regex
        3. Compare against ground truth
        4. Compute exact match accuracy
    
    Args:
        trainer: Trained Trainer instance
        test_data: Original test dataset (with question/answer)
        test_formatted: Formatted test dataset (with text)
        num_samples: Number of samples to evaluate (for speed)
    
    Returns:
        Dictionary of evaluation metrics
    """
    model = trainer.model
    model.eval()
    
    correct = 0
    total = 0
    parse_failures = 0
    results = []
    
    # Evaluate on a subset for speed (full eval can take hours on 1B model)
    eval_samples = min(num_samples, len(test_data))
    
    logger.info(f"Evaluating on {eval_samples} samples...")
    
    for i in range(eval_samples):
        # Get the question and ground truth
        question = test_data[i]["question"]
        ground_truth = test_data[i]["answer"]
        gt_answer = extract_numeric_answer(ground_truth)
        
        # Format the prompt (question only, not the answer)
        prompt = f"Question: {question}\nAnswer: Let's solve this step by step.\n"
        
        # Tokenize and generate
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,       # Greedy decoding for determinism
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = extract_numeric_answer(generated)
        
        if predicted_answer is None:
            parse_failures += 1
        
        is_correct = (predicted_answer == gt_answer) if (predicted_answer and gt_answer) else False
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "question": question[:100],
            "ground_truth": gt_answer,
            "predicted": predicted_answer,
            "correct": is_correct
        })
        
        # Progress logging
        if (i + 1) % 25 == 0:
            print(f"  Evaluated {i+1}/{eval_samples} | Running accuracy: {correct/total:.3f}")
    
    # Compute final metrics
    accuracy = correct / total if total > 0 else 0
    parse_rate = (total - parse_failures) / total if total > 0 else 0
    
    eval_metrics = {
        "exact_match_accuracy": round(accuracy, 4),
        "total_evaluated": total,
        "correct": correct,
        "parse_failures": parse_failures,
        "parse_rate": round(parse_rate, 4),
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Exact Match Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"  Answer Parse Rate:    {parse_rate:.4f}")
    print(f"  Parse Failures:       {parse_failures}")
    print(f"{'='*50}")
    
    # Save results
    with open(f"{LOG_DIR}/eval_results.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    
    with open(f"{LOG_DIR}/eval_details.json", "w") as f:
        json.dump(results[:50], f, indent=2)  # Save first 50 for inspection
    
    return eval_metrics


# Run evaluation (on 100 samples for demo speed; increase for full eval)
eval_metrics = evaluate_model(trainer, test_dataset, test_formatted, num_samples=100)


# ════════════════════════════════════════════════════════════════
# CELL 10: Save Adapter Weights & Summary
# ════════════════════════════════════════════════════════════════

def save_artifacts(model, tokenizer, eval_metrics):
    """
    Save the trained LoRA adapter weights, tokenizer, and metrics.
    
    Only the LoRA adapter weights are saved (~10-50MB), not the
    full base model (~2GB). The adapter can be merged with the base
    model later using peft.merge_and_unload() for serving.
    
    Args:
        model: Trained PEFT model
        tokenizer: Tokenizer instance
        eval_metrics: Evaluation results dictionary
    """
    adapter_path = f"{OUTPUT_DIR}/final_adapter"
    
    # Save LoRA adapter weights
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    # Save combined summary
    summary = {
        "model_name": MODEL_NAME,
        "training_config": {
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES,
            "epochs": NUM_EPOCHS,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "train_samples": TRAIN_SAMPLES,
            "test_samples": TEST_SAMPLES,
        },
        "evaluation": eval_metrics,
        "adapter_path": adapter_path,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{OUTPUT_DIR}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print adapter size
    adapter_size = sum(
        os.path.getsize(os.path.join(adapter_path, f))
        for f in os.listdir(adapter_path)
        if os.path.isfile(os.path.join(adapter_path, f))
    )
    
    print(f"\n--- Artifacts Saved ---")
    print(f"  Adapter weights: {adapter_path} ({adapter_size / 1e6:.1f} MB)")
    print(f"  Training summary: {OUTPUT_DIR}/training_summary.json")
    print(f"  Training metrics: {LOG_DIR}/training_metrics.json")
    print(f"  Eval results: {LOG_DIR}/eval_results.json")


save_artifacts(peft_model, tokenizer, eval_metrics)

print("\n✅ GSM8K Fine-Tuning Pipeline Complete!")
