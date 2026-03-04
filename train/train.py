"""
train.py — LoRA / QLoRA supervised fine-tuning for Qwen2.5-VL-7B-Instruct.

Fine-tunes the Qwen2.5-VL-7B model on the JSONL training data produced by
prepare_sft_data.py.  Uses HuggingFace TRL's SFTTrainer with PEFT LoRA
for parameter-efficient training.

Supports:
  - LoRA (16-bit base model)
  - QLoRA (4-bit quantised base model, for ≤12GB VRAM GPUs)

Usage:
  # Standard LoRA fine-tune (needs ≥24GB VRAM):
  python -m train.train --data train.jsonl

  # QLoRA 4-bit (fits on 12-16GB VRAM):
  python -m train.train --data train.jsonl --qlora

  # Custom hyperparameters:
  python -m train.train --data train.jsonl --epochs 3 --lr 2e-5 --lora-rank 64
"""

import argparse
import json
import os
import sys

from PIL import Image

import torch
from datasets import Dataset
from transformers import (
    AutoProcessor,
    Qwen3_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3.5-9B"

# Special tokens for anchor text boundaries (must match prepare_sft_data.py)
SPECIAL_TOKENS = ["<|anchor_start|>", "<|anchor_end|>"]

# LoRA target modules for Qwen2.5-VL
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_jsonl_dataset(jsonl_path: str) -> Dataset:
    """
    Load the JSONL training data and convert to HuggingFace Dataset.

    Each JSONL line has:
    {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "..."},
                {"type": "image", "image": "images/page.png"}
            ]},
            {"role": "assistant", "content": "..."}
        ],
        "metadata": {...}
    }

    Converts to the chat format expected by Qwen2.5-VL's processor.
    """
    examples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARNING: Skipping malformed JSON on line {line_num}: {e}")
                continue

            messages = record["messages"]
            user_msg = messages[0]
            assistant_msg = messages[1]

            # Extract image path and text prompt from user message
            image_path = None
            prompt_text = ""
            for part in user_msg["content"]:
                if part["type"] == "image":
                    image_path = part["image"]
                elif part["type"] == "text":
                    prompt_text = part["text"]

            if not image_path or not os.path.exists(image_path):
                print(f"  WARNING: Image not found: {image_path}, skipping line {line_num}")
                continue

            ground_truth = assistant_msg["content"]
            if not ground_truth:
                continue

            # Build the chat messages in Qwen2.5-VL format
            # Order: text first, image second (matches prepare_sft_data.py)
            formatted_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image", "image": image_path},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ground_truth},
                    ],
                },
            ]

            examples.append({
                "messages": formatted_messages,
                "images": [image_path],
            })

    print(f"Loaded {len(examples)} training examples from {jsonl_path}")
    return Dataset.from_list(examples)


# ---------------------------------------------------------------------------
# Collator for multi-modal chat
# ---------------------------------------------------------------------------

class Qwen2VLDataCollator:
    """
    Custom data collator for Qwen2.5-VL that handles image loading
    and chat template application.

    Images are expected to be pre-resized to the target resolution
    (1288px max edge) by prepare_sft_data.py.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples: list) -> dict:
        texts = []
        images_list = []

        for example in examples:
            messages = example["messages"]
            image_paths = example["images"]

            # Apply chat template to get the formatted text
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

            # Load images (already resized to 1288px by prepare_sft_data.py)
            images = []
            for img_path in image_paths:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            images_list.append(images)

        # Flatten images for processor (it expects a flat list)
        all_images = [img for imgs in images_list for img in imgs]

        # Process everything together
        batch = self.processor(
            text=texts,
            images=all_images if all_images else None,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )

        # Create labels (same as input_ids, with padding tokens masked)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Mask the user/system tokens so loss is only computed on assistant response
        # We find where the assistant tokens start and only compute loss there
        batch["labels"] = labels

        return batch


# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------

def load_model_and_processor(qlora: bool = False):
    """Load the Qwen3.5-9B model with optional 4-bit quantisation."""

    print(f"Loading model: {MODEL_ID}")
    print(f"  Mode: {'QLoRA (4-bit)' if qlora else 'LoRA (16-bit)'}")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    # Register special tokens so they are treated as atomic units
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    print(f"  Registered special tokens: {SPECIAL_TOKENS}")

    if qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen3_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = Qwen3_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable()

    model.config.use_cache = False  # Required for gradient checkpointing

    # Resize token embeddings to accommodate new special tokens
    model.resize_token_embeddings(len(processor.tokenizer))
    print(f"  Token embeddings resized to {len(processor.tokenizer)}")

    return model, processor


def apply_lora(model, rank: int = 32, alpha: int = 64, dropout: float = 0.05):
    """Apply LoRA adapters to the model."""
    print(f"  LoRA config: rank={rank}, alpha={alpha}, dropout={dropout}")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    """Main training loop."""

    # Load dataset
    print("\n" + "=" * 60)
    print("Loading dataset")
    print("=" * 60)
    dataset = load_jsonl_dataset(args.data)

    if len(dataset) == 0:
        print("ERROR: No valid training examples found.")
        sys.exit(1)

    # Optionally split for validation
    if args.val_split > 0:
        split = dataset.train_test_split(test_size=args.val_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"  Train: {len(train_dataset)}, Eval: none")

    # Load model
    print("\n" + "=" * 60)
    print("Loading model")
    print("=" * 60)
    model, processor = load_model_and_processor(qlora=args.qlora)
    model = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)

    # Data collator
    collator = Qwen2VLDataCollator(processor)

    # Training configuration
    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch" if eval_dataset else "no",
        report_to="none",
        gradient_checkpointing=True,
        max_seq_length=2048,
        dataset_text_field=None,  # We use a custom collator
        dataset_kwargs={"skip_prepare_dataset": True},
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor.tokenizer,
    )

    # Train
    trainer.train()

    # Save
    print("\n" + "=" * 60)
    print("Saving model")
    print("=" * 60)

    # Save LoRA adapters
    lora_dir = os.path.join(args.output_dir, "lora_adapters")
    model.save_pretrained(lora_dir)
    processor.save_pretrained(lora_dir)
    print(f"  LoRA adapters saved to: {lora_dir}")

    # Optionally merge and save full model
    if args.merge:
        print("  Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        merged_dir = os.path.join(args.output_dir, "merged_model")
        merged_model.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)
        print(f"  Merged model saved to: {merged_dir}")

    print("\nTraining complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3.5-9B with LoRA/QLoRA for OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # LoRA fine-tune (24GB+ VRAM):\n"
            "  python -m train.train --data train.jsonl\n\n"
            "  # QLoRA 4-bit (12-16GB VRAM):\n"
            "  python -m train.train --data train.jsonl --qlora\n\n"
            "  # Custom config:\n"
            "  python -m train.train --data train.jsonl --epochs 5 --lr 1e-5 --lora-rank 64\n"
        ),
    )

    # Data
    parser.add_argument("--data", required=True, help="Path to JSONL training data")
    parser.add_argument("--val-split", type=float, default=0.05,
                        help="Fraction of data to use for validation (default: 0.05)")

    # Model
    parser.add_argument("--qlora", action="store_true",
                        help="Use QLoRA 4-bit quantisation (fits on 12-16GB VRAM)")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (default: 32)")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="LoRA alpha (default: 64)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")

    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")

    # Output
    parser.add_argument("--output-dir", default="checkpoints/qwen2.5-vl-7b-ocr",
                        help="Output directory for checkpoints (default: checkpoints/qwen2.5-vl-7b-ocr)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA weights into base model after training")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
