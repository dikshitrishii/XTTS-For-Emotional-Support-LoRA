#!/usr/bin/env python3
"""
Fixed Perceiver Resampler LoRA fine-tuning script for XTTS-v2.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig


class PerceiverLoraDataset(Dataset):
    """Dataset for Perceiver Resampler LoRA training."""
    
    def __init__(self, csv_path: str):
        data = [l.strip().split(",") for l in Path(csv_path).read_text().splitlines()]
        self.wav_paths: List[str] = [row[0] for row in data]
        self.texts: List[str] = [",".join(row[1:]) for row in data]

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        return {
            "wav_path": self.wav_paths[idx],
            "text": self.texts[idx],
        }


def setup_perceiver_lora_training(model_ckpt: str, config_path: str) -> Xtts:
    """Initialize XTTS with Perceiver Resampler LoRA adapters."""
    cfg = XttsConfig()
    cfg.load_json(config_path)

    # Enable Perceiver LoRA in config
    cfg.model_args.use_perceiver_lora = True
    cfg.model_args.gpt_use_perceiver_resampler = True

    model: Xtts = Xtts.init_from_config(cfg)
    
    # Fix: Extract checkpoint directory and vocab path
    checkpoint_dir = os.path.dirname(model_ckpt)
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    
    # Check if vocab.json exists
    if not os.path.exists(vocab_path):
        config_dir = os.path.dirname(config_path)
        vocab_path = os.path.join(config_dir, "vocab.json")
        
        if not os.path.exists(vocab_path):
            print(f"⚠️ vocab.json not found. Using default tokenizer.")
            vocab_path = None
    
    # Load checkpoint with proper parameters
    try:
        model.load_checkpoint(
            cfg, 
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=model_ckpt,
            vocab_path=vocab_path,
            eval=False
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Trying alternative loading method...")
        model.load_checkpoint(
            cfg,
            checkpoint_path=model_ckpt,
            vocab_path=vocab_path,
            eval=False
        )

    # Debug model structure before setting training mode
    from TTS.tts.models.xtts import debug_model_structure
    debug_model_structure(model.gpt)

    # Set Perceiver LoRA training mode
    model.set_perceiver_training_mode("perceiver_lora")
    
    # Print parameter status
    model.print_perceiver_parameter_status()
    
    return model



def create_perceiver_lora_optimizer(model: Xtts, lr: float = 1e-3) -> optim.Optimizer:
    """Create optimizer for Perceiver Resampler LoRA parameters only."""
    perceiver_lora_params = model.get_perceiver_lora_parameters()
    
    if not perceiver_lora_params:
        print("❌ No Perceiver LoRA parameters found.")
        print("🔍 Available parameter names:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  - {name}")
        raise RuntimeError("❌ No Perceiver LoRA parameters found – check your integration.")
    
    print(f"🎯 Optimizing {len(perceiver_lora_params)} Perceiver LoRA parameter groups")
    
    return optim.AdamW(
        perceiver_lora_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )


def train_perceiver_step(model: Xtts,
                        batch: Dict,
                        optimizer: optim.Optimizer,
                        loss_fn: nn.Module,
                        device: torch.device) -> float:
    """Single training step for Perceiver Resampler LoRA."""
    model.train()

    # Example forward pass - adapt to your specific training pipeline
    texts = batch["texts"]
    wav_paths = batch["wav_paths"]
    
    # Placeholder loss - replace with your actual conditioning loss
    # This should involve computing conditioning latents and comparing with targets
    loss = loss_fn(torch.zeros(1, device=device), torch.zeros(1, device=device))

    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping for Perceiver LoRA parameters
    torch.nn.utils.clip_grad_norm_(model.get_perceiver_lora_parameters(), 1.0)
    
    optimizer.step()
    return loss.item()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup data
    train_ds = PerceiverLoraDataset(args.train_csv)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda batch: {
            "wav_paths": [item["wav_path"] for item in batch],
            "texts": [item["text"] for item in batch],
        }
    )

    # Setup model and optimizer
    model = setup_perceiver_lora_training(args.model_ckpt, args.config_json).to(device)
    optimizer = create_perceiver_lora_optimizer(model, lr=args.lr)
    loss_fn = nn.MSELoss()  # Replace with appropriate loss

    # Training loop
    global_step = 0
    best_loss = float("inf")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        for batch in train_loader:
            loss = train_perceiver_step(model, batch, optimizer, loss_fn, device)
            global_step += 1

            if global_step % args.log_every == 0:
                print(f"[{epoch}/{args.epochs}] step {global_step:>6}  perceiver_loss={loss:.4f}")

            # Save best Perceiver LoRA checkpoint
            if loss < best_loss:
                best_loss = loss
                ckpt_path = os.path.join(args.out_dir, "best_perceiver_lora.pth")
                model.save_perceiver_lora_checkpoint(ckpt_path)

        # Epoch-level save
        epoch_ckpt = os.path.join(args.out_dir, f"perceiver_lora_epoch{epoch}.pth")
        model.save_perceiver_lora_checkpoint(epoch_ckpt)

    print("✅ Perceiver Resampler LoRA training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perceiver Resampler LoRA fine-tuning for XTTS-v2")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="perceiver_lora_runs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    main(args)
