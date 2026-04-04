# Part 2: GSM8K Fine-Tuning with LoRA

## Overview

Fine-tunes a LLaMA 3.2 1B model on 3000 GSM8K math reasoning samples using LoRA (Low-Rank Adaptation). The script is designed to run on Google Colab with a T4 GPU.

## Quick Start (Google Colab)

1. **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com)
2. **Set GPU Runtime**: `Runtime → Change runtime type → T4 GPU`
3. **Upload the script**: Upload `gsm8k_lora_finetuning.py`
4. **Run in a notebook cell**:
   ```python
   !pip install -q torch transformers peft datasets accelerate bitsandbytes trl scipy
   %run gsm8k_lora_finetuning.py
   ```

   **Or** copy each `# CELL N` section into separate Colab cells for step-by-step execution.

## Model Access

### Option A: LLaMA 3.2 1B (Primary)
- Requires a HuggingFace account with Meta license accepted
- Login in Colab: `!huggingface-cli login --token YOUR_TOKEN`

### Option B: TinyLlama 1.1B (Automatic Fallback)
- Openly available, no special access needed
- The script automatically falls back if LLaMA is unavailable

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank (r) | 8 | Balance between capacity and efficiency |
| LoRA alpha | 16 | 2× rank — standard scaling heuristic |
| Target modules | q_proj, v_proj | Attention layers for reasoning |
| Batch size | 4 (effective: 32) | Gradient accumulation simulates larger batch |
| Epochs | 3 | Sufficient for fine-tuning convergence |
| Learning rate | 2e-4 | Standard for LoRA SFT |
| Precision | FP16 | Halves VRAM usage on T4 GPU |
| Quantization | 4-bit NF4 | QLoRA enables training on 16GB VRAM |

## Expected Output

```
Training completed in ~20-30 minutes (T4 GPU)
Exact Match Accuracy: ~25-35% (baseline zero-shot: ~5%)
Adapter size: ~10-50 MB
```

## Output Files

```
gsm8k_lora_output/
├── final_adapter/           # LoRA weights + tokenizer
├── checkpoints/             # Per-epoch checkpoints
├── logs/
│   ├── training_metrics.json
│   ├── eval_results.json
│   └── eval_details.json
└── training_summary.json    # Full config + results
```
