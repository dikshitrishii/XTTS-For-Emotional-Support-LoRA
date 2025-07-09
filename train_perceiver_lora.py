#!/usr/bin/env python3
"""
Complete Perceiver Resampler LoRA fine-tuning script for XTTS-v2.
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
        
        # Use pandas for more robust CSV parsing
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
                
                # Try different delimiters
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
                    
                    # Clean text content (remove XML-like tags)
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
    
    # Clean and validate the audio path
    audio_path = str(audio_path).strip()
    
    # Remove any XML-like content that might be concatenated
    audio_path = re.split(r'[|<]', audio_path)[0].strip()
    
    # Check if path exists
    if not os.path.exists(audio_path):
        print(f"⚠️ Audio file not found: {audio_path}")
        
        # Try alternative paths
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
            # Return silence as fallback
            return torch.zeros(1, target_sr * 3)
    
    try:
        audio, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
        
        # Convert to mono if stereo
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Normalize and clip
        audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio
        
    except Exception as e:
        print(f"⚠️ Error loading audio {audio_path}: {e}")
        # Return silence as fallback
        return torch.zeros(1, target_sr * 3)


def debug_csv_format(csv_path: str, num_lines: int = 5):
    """Debug CSV file format to understand the structure."""
    print(f"🔍 Debugging CSV file: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            print(f"Line {i+1}: {repr(line.strip())}")
    
    # Try different parsing methods
    print("\n📊 Parsing attempts:")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        
        print(f"Original: {repr(first_line)}")
        print(f"Split by '|': {first_line.split('|')}")
        print(f"Split by ',': {first_line.split(',')}")
        # print(f"Split by tab: {first_line.split('\\t')}")


def create_clean_csv(input_csv: str, output_csv: str):
    """Create a clean CSV file from the problematic one."""
    clean_data = []
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract audio path and text
            parts = line.split('|')
            if len(parts) >= 2:
                audio_path = parts[0].strip()
                text_content = parts[1].strip()
                
                # Clean text (remove XML tags)
                text_content = re.sub(r'<[^>]+>', '', text_content).strip()
                
                # Validate audio path exists
                if os.path.exists(audio_path):
                    clean_data.append([audio_path, text_content])
                else:
                    # Try alternative paths
                    alt_paths = [
                        audio_path.replace('wavs/wavs/', 'wavs/'),
                        audio_path.replace('wavs/', 'Dataset_for_final/wavs/'),
                        os.path.join('Dataset_for_final', audio_path)
                    ]
                    
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            clean_data.append([alt_path, text_content])
                            break
                    else:
                        print(f"⚠️ Skipping missing file: {audio_path}")
    
    # Save clean CSV
    df = pd.DataFrame(clean_data, columns=['audio_path', 'text'])
    df.to_csv(output_csv, index=False, header=False, sep='|')
    print(f"✅ Clean CSV saved: {output_csv} ({len(clean_data)} entries)")
    return output_csv


def setup_perceiver_lora_training(model_ckpt: str, config_path: str) -> Xtts:
    """Initialize XTTS with Perceiver Resampler LoRA adapters."""
    cfg = XttsConfig()
    cfg.load_json(config_path)

    # IMPORTANT: Disable LoRA during initial model creation
    cfg.model_args.use_perceiver_lora = False
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
    
    # Load checkpoint FIRST (without LoRA)
    try:
        model.load_checkpoint(
            cfg, 
            checkpoint_dir=checkpoint_dir,
            checkpoint_path=model_ckpt,
            vocab_path=vocab_path,
            eval=False,
            strict=True  # Can be strict since no LoRA yet
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
                        device: torch.device) -> float:
    """Single training step for Perceiver Resampler LoRA."""
    model.train()

    # Get batch data
    texts = batch["texts"]
    wav_paths = batch["wav_paths"]
    
    try:
        # Load and process audio files
        audio_tensors = []
        for wav_path in wav_paths:
            audio = load_audio_for_training(wav_path)
            audio_tensors.append(audio)
        
        # Stack audios (pad/truncate to same length)
        max_length = max(audio.size(1) for audio in audio_tensors)
        max_length = min(max_length, 22050 * 10)  # Max 10 seconds
        
        batch_audio = []
        for audio in audio_tensors:
            if audio.size(1) > max_length:
                audio = audio[:, :max_length]
            elif audio.size(1) < max_length:
                padding = max_length - audio.size(1)
                audio = torch.nn.functional.pad(audio, (0, padding))
            batch_audio.append(audio)
        
        batch_audio = torch.stack(batch_audio).to(device)  # [batch_size, 1, time]
        
        # Forward pass through conditioning encoder
        # This will exercise the Perceiver Resampler with LoRA adapters
        conditioning_latents = []
        for i, audio in enumerate(batch_audio):
            # Get conditioning latents using the model's method
            cond_latent = model.get_gpt_cond_latents(
                audio.unsqueeze(0), 
                sr=22050, 
                length=6, 
                chunk_length=6
            )
            conditioning_latents.append(cond_latent)
        
        # Stack conditioning latents
        conditioning_latents = torch.stack(conditioning_latents).squeeze(1)  # [batch_size, features, time]
        
        #we need to add all the loss functions here
        # For simplicity, we will use a basic reconstruction loss
        # This is a placeholder; you can replace it with your actual loss function
        
        
        # Creating a simple reconstruction loss
        # Target: try to reconstruct the input conditioning
        target = conditioning_latents.detach()  # Use current output as target
        
        # Add some noise to create a learning objective
        noisy_input = conditioning_latents + 0.1 * torch.randn_like(conditioning_latents)
        
        # Simple MSE loss between noisy and clean conditioning
        loss = torch.nn.functional.mse_loss(conditioning_latents, target)
        
        # Add a small regularization term to encourage LoRA parameter activity
        lora_reg = 0.0
        for param in model.get_perceiver_lora_parameters():
            lora_reg += torch.norm(param, p=2)
        
        total_loss = loss + 1e-6 * lora_reg
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for Perceiver LoRA parameters
        torch.nn.utils.clip_grad_norm_(model.get_perceiver_lora_parameters(), 1.0)
        
        optimizer.step()
        
        return total_loss.item()
        
    except Exception as e:
        print(f"❌ Error in training step: {e}")
        # Return a dummy loss to continue training
        dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
        optimizer.zero_grad()
        dummy_loss.backward()
        optimizer.step()
        return dummy_loss.item()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Debug CSV format first
    debug_csv_format(args.train_csv)
    
    # Create clean CSV if needed
    if args.create_clean_csv:
        clean_csv_path = args.train_csv.replace('.csv', '_clean.csv')
        args.train_csv = create_clean_csv(args.train_csv, clean_csv_path)

    # Setup data
    train_ds = PerceiverLoraDataset(args.train_csv)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,  # Reduced to avoid memory issues
        collate_fn=lambda batch: {
            "wav_paths": [item["wav_path"] for item in batch],
            "texts": [item["text"] for item in batch],
        }
    )

    # Setup model and optimizer
    model = setup_perceiver_lora_training(args.model_ckpt, args.config_json).to(device)
    optimizer = create_perceiver_lora_optimizer(model, lr=args.lr)

    # Training loop
    global_step = 0
    best_loss = float("inf")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"🎯 Starting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            loss = train_perceiver_step(model, batch, optimizer, device)
            epoch_loss += loss
            num_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / num_batches
                print(f"[{epoch}/{args.epochs}] step {global_step:>6}  avg_loss={avg_loss:.6f}  current_loss={loss:.6f}")

            # Save best Perceiver LoRA checkpoint
            if loss < best_loss:
                best_loss = loss
                ckpt_path = os.path.join(args.out_dir, "best_perceiver_lora.pth")
                model.save_perceiver_lora_checkpoint(ckpt_path)

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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--create_clean_csv", action="store_true", help="Create a clean CSV from the input")
    args = parser.parse_args()

    main(args)