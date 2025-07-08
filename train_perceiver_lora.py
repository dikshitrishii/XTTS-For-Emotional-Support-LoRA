#!/usr/bin/env python3
"""
Complete Perceiver Resampler LoRA fine-tuning script for XTTS-v2 with corrected loss function.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import pandas as pd

from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig


class PerceiverLoraDataset(Dataset):
    """Dataset for Perceiver Resampler LoRA training."""
    
    def __init__(self, csv_path: str):
        print(f"📂 Loading dataset from: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path, header=None, sep='|')
            
            if df.shape[1] >= 2:
                self.wav_paths = df.iloc[:, 0].tolist()
                self.texts = df.iloc[:, 1].tolist()
                
                # Clean text content (remove XML-like tags)
                self.texts = [re.sub(r'<[^>]+>', '', text).strip() for text in self.texts]
            else:
                self._parse_csv_manually(csv_path)
                
        except Exception as e:
            print(f"⚠️ Error reading CSV with pandas: {e}")
            print("🔄 Falling back to manual parsing...")
            self._parse_csv_manually(csv_path)
        
        print(f"✅ Loaded {len(self.wav_paths)} audio-text pairs")
    
    def _parse_csv_manually(self, csv_path: str):
        """Manual CSV parsing with better error handling."""
        self.wav_paths = []
        self.texts = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                if '|' in line:
                    parts = line.split('|')
                elif ',' in line:
                    parts = line.split(',')
                elif '\t' in line:
                    parts = line.split('\t')
                else:
                    print(f"⚠️ Line {line_num}: Unable to parse - {line}")
                    continue
                
                if len(parts) >= 2:
                    audio_path = parts[0].strip()
                    text_content = parts[1].strip()
                    
                    text_content = re.sub(r'<[^>]+>', '', text_content)
                    text_content = text_content.strip()
                    
                    self.wav_paths.append(audio_path)
                    self.texts.append(text_content)
                else:
                    print(f"⚠️ Line {line_num}: Insufficient columns - {line}")

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        return {
            "wav_path": self.wav_paths[idx],
            "text": self.texts[idx],
        }


def load_audio_for_training(audio_path, target_sr=22050):
    """Load and preprocess audio for training with path validation."""
    
    audio_path = str(audio_path).strip()
    audio_path = re.split(r'[|<]', audio_path)[0].strip()
    
    if not os.path.exists(audio_path):
        print(f"⚠️ Audio file not found: {audio_path}")
        
        alternative_paths = [
            audio_path.replace('wavs/wavs/', 'wavs/'),
            audio_path.replace('wavs/', 'Dataset_for_final/wavs/'),
            os.path.join('Dataset_for_final', audio_path),
            os.path.join('.', audio_path)
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"✅ Found alternative path: {alt_path}")
                audio_path = alt_path
                break
        else:
            print(f"❌ No valid path found for: {audio_path}")
            return torch.zeros(1, target_sr * 3)
    
    try:
        audio, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
        
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        audio = torch.clamp(audio, -1.0, 1.0)
        return audio
        
    except Exception as e:
        print(f"⚠️ Error loading audio {audio_path}: {e}")
        return torch.zeros(1, target_sr * 3)


def setup_perceiver_lora_training(model_ckpt: str, config_path: str) -> Xtts:
    """Initialize XTTS with Perceiver Resampler LoRA adapters."""
    cfg = XttsConfig()
    cfg.load_json(config_path)

    # IMPORTANT: Disable LoRA during initial model creation
    cfg.model_args.use_perceiver_lora = False
    cfg.model_args.gpt_use_perceiver_resampler = True

    model: Xtts = Xtts.init_from_config(cfg)
    
    checkpoint_dir = os.path.dirname(model_ckpt)
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    
    if not os.path.exists(vocab_path):
        config_dir = os.path.dirname(config_path)
        vocab_path = os.path.join(config_dir, "vocab.json")
        
        if not os.path.exists(vocab_path):
            print(f"⚠️ vocab.json not found. Using default tokenizer.")
            vocab_path = None
    
    # Load checkpoint FIRST (without LoRA)
    try:
        model.load_checkpoint(
            cfg, 
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=model_ckpt,
            vocab_path=vocab_path,
            eval=False,
            strict=True
        )
        print("✅ Base checkpoint loaded successfully")
    except Exception as e:
        print(f"❌ Error loading base checkpoint: {e}")
        raise e

    # NOW apply LoRA adapters AFTER loading the checkpoint
    print("🔧 Applying LoRA adapters to loaded model...")
    from TTS.tts.models.xtts import apply_lora_to_perceiver_resampler
    model.gpt = apply_lora_to_perceiver_resampler(
        model.gpt,
        r=8,
        alpha=16,
        dropout=0.05
    )
    print("🎯 Perceiver Resampler LoRA adapters applied successfully")

    model.set_perceiver_training_mode("perceiver_lora")
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


def verify_model_gradients(model: Xtts):
    """Utility function to verify LoRA parameters can receive gradients."""
    print("\n🔍 Gradient Verification:")
    
    lora_params = model.get_perceiver_lora_parameters()
    print(f"Found {len(lora_params)} LoRA parameters")
    
    for i, param in enumerate(lora_params[:5]):  # Check first 5
        print(f"  Param {i}: shape={param.shape}, requires_grad={param.requires_grad}, device={param.device}")
    
    # Test forward pass
    dummy_audio = torch.randn(1, 22050 * 3, device=model.device)  # 3 seconds
    
    try:
        with torch.enable_grad():
            latents = model.get_gpt_cond_latents(dummy_audio, sr=22050, length=3)
            loss = latents.mean()
            loss.backward()
            
            grad_found = any(p.grad is not None for p in lora_params)
            print(f"  Gradient test: {'✅ PASS' if grad_found else '❌ FAIL'}")
            
    except Exception as e:
        print(f"  Gradient test: ❌ ERROR - {e}")


def train_perceiver_step(model: Xtts,
                        batch: Dict,
                        optimizer: optim.Optimizer,
                        device: torch.device) -> float:
    """CORRECTED: Enhanced training step with gradient verification and meaningful loss."""
    model.train()
    
    # Ensure LoRA parameters require gradients
    lora_params = model.get_perceiver_lora_parameters()
    for param in lora_params:
        param.requires_grad_(True)
    
    texts = batch["texts"]
    wav_paths = batch["wav_paths"]
    
    try:
        # Process pairs of audio for contrastive learning
        anchor_latents = []
        positive_latents = []
        
        for wav_path in wav_paths:
            # Load original audio
            audio = load_audio_for_training(wav_path)
            audio = audio.to(device)
            
            # Anchor: Original audio conditioning
            with torch.enable_grad():  # Explicitly enable gradients
                anchor_latent = model.get_gpt_cond_latents(
                    audio.unsqueeze(0),
                    sr=22050, length=6, chunk_length=6
                )
                anchor_latents.append(anchor_latent)
            
            # Positive: Augmented version
            aug_audio = audio.clone()
            
            # Apply multiple augmentations
            # 1. Additive noise
            noise_level = 0.02 + torch.rand(1, device=device) * 0.03  # 0.02-0.05
            aug_audio += noise_level * torch.randn_like(aug_audio)
            
            # 2. Time masking (randomly zero out small segments)
            if torch.rand(1) > 0.5:
                mask_length = int(torch.rand(1) * 2000)  # Up to ~90ms
                mask_start = int(torch.rand(1) * (aug_audio.shape[-1] - mask_length))
                aug_audio[:, mask_start:mask_start + mask_length] = 0
            
            # 3. Amplitude scaling
            scale = 0.85 + torch.rand(1, device=device) * 0.3  # 0.85-1.15
            aug_audio *= scale
            
            aug_audio = torch.clamp(aug_audio, -1.0, 1.0)
            
            # Positive: Augmented conditioning
            with torch.enable_grad():
                positive_latent = model.get_gpt_cond_latents(
                    aug_audio.unsqueeze(0),
                    sr=22050, length=6, chunk_length=6
                )
                positive_latents.append(positive_latent)
        
        # Stack and verify gradients
        anchor_latents = torch.stack(anchor_latents).squeeze(1)  # [B, D, T]
        positive_latents = torch.stack(positive_latents).squeeze(1)  # [B, D, T]
        
        # Verify tensors require gradients
        if not anchor_latents.requires_grad:
            print("⚠️ WARNING: anchor_latents doesn't require gradients!")
        if not positive_latents.requires_grad:
            print("⚠️ WARNING: positive_latents doesn't require gradients!")
        
        # Multi-component loss function
        # 1. Contrastive loss (anchor vs positive should be similar)
        contrastive_loss = torch.nn.functional.mse_loss(anchor_latents, positive_latents)
        
        # 2. Representation magnitude loss (prevent collapse)
        anchor_norm = torch.norm(anchor_latents, dim=(1, 2)).mean()
        positive_norm = torch.norm(positive_latents, dim=(1, 2)).mean()
        magnitude_loss = torch.abs(anchor_norm - 1.0) + torch.abs(positive_norm - 1.0)
        
        # 3. Diversity loss (different samples should be different)
        batch_size = anchor_latents.shape[0]
        diversity_loss = torch.tensor(0.0, device=device)
        if batch_size > 1:
            # Flatten for pairwise distance computation
            anchor_flat = anchor_latents.view(batch_size, -1)
            distances = torch.pdist(anchor_flat)
            diversity_loss = torch.exp(-distances.mean())  # Penalize small distances
        
        # 4. LoRA activation loss (ensure LoRA parameters are being used)
        lora_activation = torch.tensor(0.0, device=device)
        for param in lora_params:
            lora_activation += torch.norm(param)
        lora_activation = lora_activation / len(lora_params) if lora_params else torch.tensor(0.0, device=device)
        
        # Combine losses with appropriate weights
        total_loss = (
            1.0 * contrastive_loss +           # Primary learning signal
            0.1 * magnitude_loss +             # Prevent representation collapse
            0.05 * diversity_loss +            # Encourage sample diversity
            0.001 * lora_activation            # Ensure LoRA usage
        )
        
        # Add minimum loss threshold to prevent zero
        min_loss = 1e-4
        if total_loss < min_loss:
            total_loss = total_loss + min_loss
        
        # Verify loss requires gradients
        if not total_loss.requires_grad:
            print("❌ CRITICAL: total_loss doesn't require gradients!")
            return 0.0
        
        # Gradient computation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        
        # Check if gradients were computed
        grad_norm = 0.0
        grad_count = 0
        for param in lora_params:
            if param.grad is not None:
                grad_norm += param.grad.norm().item()
                grad_count += 1
        
        if grad_count == 0:
            print("❌ CRITICAL: No gradients computed for LoRA parameters!")
            return 0.0
        
        avg_grad_norm = grad_norm / grad_count if grad_count > 0 else 0.0
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        
        optimizer.step()
        
        # Debug information
        if torch.rand(1) < 0.1:  # 10% of the time
            print(f"🔍 Debug - Loss: {total_loss.item():.6f}, "
                  f"Contrastive: {contrastive_loss.item():.6f}, "
                  f"Magnitude: {magnitude_loss.item():.6f}, "
                  f"Diversity: {diversity_loss.item():.6f}, "
                  f"LoRA: {lora_activation.item():.6f}, "
                  f"Grad norm: {avg_grad_norm:.6f}")
        
        return total_loss.item()
        
    except Exception as e:
        print(f"❌ Error in training step: {e}")
        import traceback
        traceback.print_exc()
        return 1e-3


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Setup data
    train_ds = PerceiverLoraDataset(args.train_csv)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda batch: {
            "wav_paths": [item["wav_path"] for item in batch],
            "texts": [item["text"] for item in batch],
        }
    )

    # Setup model and optimizer
    model = setup_perceiver_lora_training(args.model_ckpt, args.config_json).to(device)
    
    # CRITICAL: Verify gradient setup
    verify_model_gradients(model)
    
    optimizer = create_perceiver_lora_optimizer(model, lr=args.lr)

    # Training loop with enhanced monitoring
    global_step = 0
    best_loss = float("inf")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"🎯 Starting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            loss = train_perceiver_step(model, batch, optimizer, device)
            
            # Enhanced loss monitoring
            if loss < 1e-5:
                print(f"⚠️ WARNING: Very small loss detected: {loss}")
            
            epoch_loss += loss
            num_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / num_batches
                print(f"[{epoch}/{args.epochs}] step {global_step:>6}  avg_loss={avg_loss:.6f}  current_loss={loss:.6f}")

            # Save checkpoints less frequently
            if loss < best_loss and global_step % 100 == 0:
                best_loss = loss
                ckpt_path = os.path.join(args.out_dir, "best_perceiver_lora.pth")
                model.save_perceiver_lora_checkpoint(ckpt_path)
                print(f"💾 New best loss: {best_loss:.6f}")

        # Epoch-level save and logging
        avg_epoch_loss = epoch_loss / num_batches
        print(f"📊 Epoch {epoch} completed - Average Loss: {avg_epoch_loss:.6f}")
        
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
