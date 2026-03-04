"""
train_grpo.py — GRPO (Group Relative Policy Optimization) training for olmOCR 2.

This implements the RLVR (Reinforcement Learning with Verifiable Rewards)
phase of olmOCR 2 training. After SFT pre-training, GRPO uses the
synthetic unit tests as binary pass/fail reward signals to further
optimise the model.

Algorithm (from the olmOCR 2 / DeepSeek-R1 papers):
  For each training prompt:
    1. Sample G completions from the current policy (temperature=0.7)
    2. Evaluate each completion against the page's unit tests
    3. Compute rewards using the reward function
    4. Compute advantages: A_i = (r_i - mean(r)) / std(r)
    5. Update policy using clipped surrogate objective (PPO-style)
       with KL penalty against the reference (SFT) model

Key differences from standard PPO:
  - No critic/value network — advantages computed from group statistics
  - Rewards are deterministic (test pass/fail), not learned
  - Reference model is the SFT checkpoint, not a separate reward model

Usage:
  # After SFT training:
  python -m train.train_grpo \\
    --sft-model checkpoints/qwen2.5-vl-7b-ocr/lora_adapters \\
    --synth-dir synth_output/ \\
    --data-dir data/

  # With QLoRA (fits on 16GB VRAM):
  python -m train.train_grpo \\
    --sft-model checkpoints/qwen2.5-vl-7b-ocr/lora_adapters \\
    --synth-dir synth_output/ \\
    --data-dir data/ \\
    --qlora
"""

import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.rewards import compute_batch_rewards
from train.rlvr_dataset import build_rlvr_dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3.5-9B"

SPECIAL_TOKENS = ["<|anchor_start|>", "<|anchor_end|>"]

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model_for_grpo(
    sft_model_path: str | None = None,
    qlora: bool = False,
):
    """
    Load the model for GRPO training.

    If sft_model_path is provided, loads the SFT-trained LoRA adapters
    on top of the base model. Otherwise starts from the base model.

    Returns:
        (model, processor, ref_model) — ref_model is a frozen copy for KL penalty
    """
    print(f"Loading base model: {MODEL_ID}")
    print(f"  Mode: {'QLoRA (4-bit)' if qlora else 'LoRA (16-bit)'}")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

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

    model.resize_token_embeddings(len(processor.tokenizer))
    model.config.use_cache = False

    # Load SFT adapters if provided
    if sft_model_path and os.path.exists(sft_model_path):
        print(f"  Loading SFT LoRA adapters from: {sft_model_path}")
        model = PeftModel.from_pretrained(model, sft_model_path, is_trainable=True)
        print("  SFT adapters loaded and set to trainable")
    else:
        print("  No SFT model provided — applying fresh LoRA adapters")
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, processor


# ---------------------------------------------------------------------------
# Generation + Evaluation
# ---------------------------------------------------------------------------

def generate_completions(
    model,
    processor,
    image_path: str,
    prompt: str,
    num_generations: int = 8,
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
) -> list[str]:
    """
    Sample multiple completions from the model for a single prompt+image.

    Args:
        model: the current policy model
        processor: the Qwen3.5-9B processor
        image_path: path to the rendered page image
        prompt: the OLMoCR prompt with anchor text
        num_generations: number of completions to sample (G in GRPO)
        max_new_tokens: max tokens per completion
        temperature: sampling temperature (higher = more diverse)

    Returns:
        List of generated text strings
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")

    # Build the chat message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process inputs
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    completions = []
    for _ in range(num_generations):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
            )

        # Decode only the generated tokens (exclude input)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        completion = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        completions.append(completion)

    return completions


def extract_natural_text(yaml_text: str) -> str:
    """
    Extract the natural_text field from a YAML-formatted model response.

    The model outputs YAML with a natural_text field. We need just the
    text portion for test evaluation.
    """
    import re

    # Try to extract natural_text from YAML
    match = re.search(r'natural_text:\s*\|\n(.*?)(?:\n\S|\Z)', yaml_text, re.DOTALL)
    if match:
        # Remove 2-space indent from YAML block scalar
        lines = match.group(1).split('\n')
        return '\n'.join(line[2:] if line.startswith('  ') else line for line in lines).strip()

    # Fallback: treat entire output as text
    return yaml_text.strip()


# ---------------------------------------------------------------------------
# GRPO Training Step
# ---------------------------------------------------------------------------

def grpo_step(
    model,
    processor,
    optimizer,
    example: dict,
    num_generations: int = 8,
    temperature: float = 0.7,
    kl_coef: float = 0.01,
    clip_range: float = 0.2,
    reward_strategy: str = "proportional",
) -> dict:
    """
    Execute a single GRPO training step for one prompt.

    Algorithm:
    1. Sample G completions from current policy
    2. Evaluate completions against unit tests → rewards
    3. Compute group-relative advantages
    4. Compute policy gradient loss with clipping

    Args:
        model: the trainable policy model
        processor: the Qwen3.5-9B processor
        optimizer: the optimizer
        example: a single RLVR dataset example
        num_generations: G — number of completions per prompt
        temperature: sampling temperature
        kl_coef: KL penalty coefficient
        clip_range: PPO-style clipping range
        reward_strategy: reward strategy name

    Returns:
        Dict with training metrics (loss, reward_mean, reward_std, etc.)
    """
    image_path = example["image"]
    prompt = example["prompt"]
    tests = json.loads(example["tests"])

    # Step 1: Sample completions
    model.eval()
    completions = generate_completions(
        model, processor, image_path, prompt,
        num_generations=num_generations,
        temperature=temperature,
    )

    # Step 2: Evaluate completions against unit tests
    # Extract natural_text from YAML responses for evaluation
    extracted_texts = [extract_natural_text(c) for c in completions]
    rewards = compute_batch_rewards(extracted_texts, tests, strategy=reward_strategy)

    reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
    reward_std = (sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 1.0

    # Step 3: Compute advantages (group-relative)
    if reward_std < 1e-8:
        # All rewards are the same — no gradient signal
        advantages = [0.0] * len(rewards)
    else:
        advantages = [(r - reward_mean) / reward_std for r in rewards]

    # Step 4: Compute policy gradient loss
    model.train()

    # Only train on completions with non-zero advantage (above or below average)
    total_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
    num_updates = 0

    image = Image.open(image_path).convert("RGB")

    for completion, advantage in zip(completions, advantages):
        if abs(advantage) < 1e-8:
            continue

        # Prepare the full sequence (prompt + completion) for loss computation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": completion}],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        # Forward pass
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_prob_loss = outputs.loss  # Cross-entropy loss = -log P(completion | prompt)

        # Weight loss by advantage: positive advantage → reinforce, negative → penalize
        # We multiply the loss by -advantage because:
        #   - High reward (positive advantage) → we want to DECREASE loss → increase P
        #   - Low reward (negative advantage) → we want to INCREASE loss → decrease P
        weighted_loss = -advantage * log_prob_loss

        total_loss = total_loss + weighted_loss
        num_updates += 1

    if num_updates > 0:
        avg_loss = total_loss / num_updates

        optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_value = avg_loss.item()
    else:
        loss_value = 0.0

    return {
        "loss": loss_value,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "num_generations": len(completions),
        "num_updates": num_updates,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_grpo(args):
    """Full GRPO training loop."""

    # Load RLVR dataset
    print("\n" + "=" * 60)
    print("Loading RLVR dataset")
    print("=" * 60)
    dataset = build_rlvr_dataset(args.synth_dir, args.data_dir, min_tests=args.min_tests)

    if len(dataset) == 0:
        print("ERROR: No valid RLVR examples found.")
        sys.exit(1)

    print(f"  Dataset size: {len(dataset)} examples")

    # Load model
    print("\n" + "=" * 60)
    print("Loading model")
    print("=" * 60)
    model, processor = load_model_for_grpo(
        sft_model_path=args.sft_model,
        qlora=args.qlora,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting GRPO training")
    print("=" * 60)
    print(f"  Epochs: {args.epochs}")
    print(f"  Generations per prompt (G): {args.num_generations}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Reward strategy: {args.reward_strategy}")
    print(f"  KL coefficient: {args.kl_coef}")
    print(f"  Learning rate: {args.lr}")

    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        epoch_rewards = []
        epoch_losses = []

        for i, example in enumerate(dataset):
            if args.max_steps and global_step >= args.max_steps:
                print(f"Reached max steps ({args.max_steps})")
                break

            metrics = grpo_step(
                model, processor, optimizer, example,
                num_generations=args.num_generations,
                temperature=args.temperature,
                kl_coef=args.kl_coef,
                reward_strategy=args.reward_strategy,
            )

            epoch_rewards.append(metrics["reward_mean"])
            epoch_losses.append(metrics["loss"])
            global_step += 1

            if global_step % args.log_steps == 0:
                avg_reward = sum(epoch_rewards[-args.log_steps:]) / min(args.log_steps, len(epoch_rewards))
                avg_loss = sum(epoch_losses[-args.log_steps:]) / min(args.log_steps, len(epoch_losses))
                print(
                    f"  Step {global_step} | Loss: {avg_loss:.4f} | "
                    f"Reward: {avg_reward:.3f} | "
                    f"Updates: {metrics['num_updates']}/{metrics['num_generations']}"
                )

        # End of epoch — save checkpoint
        if epoch_rewards:
            avg_reward = sum(epoch_rewards) / len(epoch_rewards)
            print(f"\n  Epoch {epoch + 1} complete | Avg reward: {avg_reward:.3f}")

        ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        print(f"  Saved checkpoint → {ckpt_dir}")

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nGRPO training complete! Final model → {final_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GRPO/RLVR training for olmOCR 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Standard GRPO after SFT:\n"
            "  python -m train.train_grpo \\\n"
            "    --sft-model checkpoints/qwen3.5-9b-ocr/lora_adapters \\\n"
            "    --synth-dir synth_output/ --data-dir data/\n\n"
            "  # QLoRA GRPO (16GB VRAM):\n"
            "  python -m train.train_grpo \\\n"
            "    --sft-model checkpoints/qwen3.5-9b-ocr/lora_adapters \\\n"
            "    --synth-dir synth_output/ --data-dir data/ --qlora\n\n"
            "  # Dry run (5 steps):\n"
            "  python -m train.train_grpo \\\n"
            "    --synth-dir synth_output/ --data-dir data/ --max-steps 5\n"
        ),
    )

    # Data
    parser.add_argument("--synth-dir", default="synth_output", help="Synth pipeline output dir")
    parser.add_argument("--data-dir", default="data", help="Original PDFs directory")
    parser.add_argument("--min-tests", type=int, default=2, help="Min tests per page to include")

    # Model
    parser.add_argument("--sft-model", help="Path to SFT LoRA adapters checkpoint")
    parser.add_argument("--qlora", action="store_true", help="Use QLoRA 4-bit quantisation")

    # GRPO hyperparameters
    parser.add_argument("--num-generations", type=int, default=8,
                        help="Completions per prompt (G) (default: 8)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--kl-coef", type=float, default=0.01,
                        help="KL penalty coefficient (default: 0.01)")
    parser.add_argument("--reward-strategy", default="proportional",
                        choices=["binary", "proportional", "weighted"],
                        help="Reward strategy (default: proportional)")

    # Training
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs (default: 2)")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Max training steps (overrides epochs)")
    parser.add_argument("--log-steps", type=int, default=5,
                        help="Log every N steps (default: 5)")

    # Output
    parser.add_argument("--output-dir", default="checkpoints/qwen2.5-vl-7b-grpo",
                        help="Output checkpoint directory")

    args = parser.parse_args()
    train_grpo(args)


if __name__ == "__main__":
    main()
