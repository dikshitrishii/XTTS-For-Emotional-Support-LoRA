import os
from dataclasses import dataclass
import pandas as pd
import librosa
import torch
import torch.nn.functional as F
import torchaudio
from coqpit import Coqpit

from TTS.tts.layers.xtts.gpt import GPT
from TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder
from TTS.tts.layers.xtts.stream_generator import init_stream_support
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, split_sentence
from TTS.tts.layers.xtts.xtts_manager import SpeakerManager, LanguageManager
from TTS.tts.models.base_tts import BaseTTS
from TTS.utils.io import load_fsspec

init_stream_support()

# LoRA Implementation for Perceiver Resampler
class PerceiverLoRALinear(torch.nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r if r > 0 else 1
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        
        if r > 0:
            self.lora_A = torch.nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = torch.nn.Parameter(torch.zeros(out_features, r))
            torch.nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            torch.nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        result = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora = self.dropout(x) @ self.lora_A.t()
            lora = lora @ self.lora_B.t()
            result = result + self.scale * lora
        return result


def apply_lora_to_perceiver_resampler(model, r=8, alpha=16, dropout=0.05):
    """Apply LoRA specifically to Perceiver Resampler attention layers."""
    
    def find_and_replace_linear_layers(module, module_name="", depth=0):
        """Recursively find and replace linear layers in Perceiver components."""
        replaced_count = 0
        
        for name, child in module.named_children():
            full_name = f"{module_name}.{name}" if module_name else name
            
            # Check if this is a Perceiver-related component
            if any(keyword in full_name.lower() for keyword in ["perceiver", "resampler", "conditioning"]):
                print(f"🔍 Exploring Perceiver component: {full_name} (depth: {depth})")
                
                # If it's a Linear layer, replace it
                if isinstance(child, torch.nn.Linear):
                    print(f"🔧 Applying LoRA to Linear layer: {full_name}")
                    lora_linear = PerceiverLoRALinear(
                        child.in_features,
                        child.out_features,
                        r=r, alpha=alpha, dropout=dropout
                    )
                    lora_linear.weight.data = child.weight.data.clone()
                    if child.bias is not None:
                        lora_linear.bias.data = child.bias.data.clone()
                    setattr(module, name, lora_linear)
                    replaced_count += 1
                
                # Recursively explore child modules
                elif hasattr(child, 'named_children'):
                    replaced_count += find_and_replace_linear_layers(child, full_name, depth + 1)
            
            # Also check for attention-related modules even outside Perceiver
            elif any(keyword in name.lower() for keyword in ["attn", "attention", "query", "key", "value"]):
                if isinstance(child, torch.nn.Linear):
                    print(f"🔧 Applying LoRA to attention layer: {full_name}")
                    lora_linear = PerceiverLoRALinear(
                        child.in_features,
                        child.out_features,
                        r=r, alpha=alpha, dropout=dropout
                    )
                    lora_linear.weight.data = child.weight.data.clone()
                    if child.bias is not None:
                        lora_linear.bias.data = child.bias.data.clone()
                    setattr(module, name, lora_linear)
                    replaced_count += 1
                elif hasattr(child, 'named_children'):
                    replaced_count += find_and_replace_linear_layers(child, full_name, depth + 1)
        
        return replaced_count
    
    # Start the replacement process
    total_replaced = find_and_replace_linear_layers(model)
    print(f"🎯 Total LoRA layers applied: {total_replaced}")
    
    return model


def debug_model_structure(model, target_keywords=["perceiver", "resampler", "conditioning"]):
    """Debug function to explore model structure."""
    print("\n🔍 Model Structure Analysis:")
    
    def explore_module(module, name="", depth=0):
        indent = "  " * depth
        print(f"{indent}{name}: {type(module).__name__}")
        
        if any(keyword in name.lower() for keyword in target_keywords):
            print(f"{indent}🎯 FOUND TARGET: {name}")
            
            # List all children
            for child_name, child in module.named_children():
                child_full_name = f"{name}.{child_name}" if name else child_name
                if isinstance(child, torch.nn.Linear):
                    print(f"{indent}  📍 Linear layer: {child_full_name} ({child.in_features} -> {child.out_features})")
                else:
                    explore_module(child, child_full_name, depth + 1)
        
        elif depth < 3:  # Limit depth to avoid too much output
            for child_name, child in module.named_children():
                child_full_name = f"{name}.{child_name}" if name else child_name
                explore_module(child, child_full_name, depth + 1)
    
    explore_module(model)


def wav_to_mel_cloning(
    wav,
    mel_norms_file="../experiments/clips_mel_norms.pth",
    mel_norms=None,
    device=torch.device("cpu"),
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    power=2,
    normalized=False,
    sample_rate=22050,
    f_min=0,
    f_max=8000,
    n_mels=80,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=normalized,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        norm="slaney",
    ).to(device)
    wav = wav.to(device)
    mel = mel_stft(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    if mel_norms is None:
        mel_norms = torch.load(mel_norms_file, map_location=device)
    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel


def load_audio(audiopath, sampling_rate):
    audio, lsr = torchaudio.load(audiopath)
    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
    if torch.any(audio > 10) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    audio.clip_(-1, 1)
    return audio


def pad_or_truncate(t, length):
    tp = t[..., :length]
    if t.shape[-1] == length:
        tp = t
    elif t.shape[-1] < length:
        tp = F.pad(t, (0, length - t.shape[-1]))
    return tp


@dataclass
class XttsAudioConfig(Coqpit):
    sample_rate: int = 22050
    output_sample_rate: int = 24000


@dataclass
class XttsArgs(Coqpit):
    gpt_batch_size: int = 1
    enable_redaction: bool = False
    kv_cache: bool = True
    gpt_checkpoint: str = None
    clvp_checkpoint: str = None
    decoder_checkpoint: str = None
    num_chars: int = 255

    # XTTS GPT Encoder params
    tokenizer_file: str = ""
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    gpt_number_text_tokens: int = None
    gpt_start_text_token: int = None
    gpt_stop_text_token: int = None
    gpt_num_audio_tokens: int = 8194
    gpt_start_audio_token: int = 8192
    gpt_stop_audio_token: int = 8193
    gpt_code_stride_len: int = 1024
    gpt_use_masking_gt_prompt_approach: bool = True
    gpt_use_perceiver_resampler: bool = True

    # HifiGAN Decoder params
    input_sample_rate: int = 22050
    output_sample_rate: int = 24000
    output_hop_length: int = 256
    decoder_input_dim: int = 1024
    d_vector_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True

    # Perceiver LoRA configuration
    use_perceiver_lora: bool = False  # Default False for normal loading
    perceiver_lora_rank: int = 8
    perceiver_lora_alpha: int = 16
    perceiver_lora_dropout: float = 0.05

    duration_const: int = 102400


class Xtts(BaseTTS):
    """ⓍTTS model implementation with Perceiver Resampler LoRA support."""

    def __init__(self, config: Coqpit):
        super().__init__(config, ap=None, tokenizer=None)
        self.mel_stats_path = None
        self.config = config
        self.gpt_checkpoint = self.args.gpt_checkpoint
        self.decoder_checkpoint = self.args.decoder_checkpoint
        self.models_dir = config.model_dir
        self.gpt_batch_size = self.args.gpt_batch_size
        self._training_mode = "inference"

        self.tokenizer = VoiceBpeTokenizer()
        self.gpt = None
        self.init_models()
        self.register_buffer("mel_stats", torch.ones(80))

    def init_models(self):
        """Initialize models with optional Perceiver Resampler LoRA integration."""
        if self.tokenizer.tokenizer is not None:
            self.args.gpt_number_text_tokens = self.tokenizer.get_number_tokens()
            self.args.gpt_start_text_token = self.tokenizer.tokenizer.token_to_id("[START]")
            self.args.gpt_stop_text_token = self.tokenizer.tokenizer.token_to_id("[STOP]")

        if self.args.gpt_number_text_tokens:
            self.gpt = GPT(
                layers=self.args.gpt_layers,
                model_dim=self.args.gpt_n_model_channels,
                start_text_token=self.args.gpt_start_text_token,
                stop_text_token=self.args.gpt_stop_text_token,
                heads=self.args.gpt_n_heads,
                max_text_tokens=self.args.gpt_max_text_tokens,
                max_mel_tokens=self.args.gpt_max_audio_tokens,
                max_prompt_tokens=self.args.gpt_max_prompt_tokens,
                number_text_tokens=self.args.gpt_number_text_tokens,
                num_audio_tokens=self.args.gpt_num_audio_tokens,
                start_audio_token=self.args.gpt_start_audio_token,
                stop_audio_token=self.args.gpt_stop_audio_token,
                use_perceiver_resampler=self.args.gpt_use_perceiver_resampler,
                code_stride_len=self.args.gpt_code_stride_len,
            )

        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=self.args.input_sample_rate,
            output_sample_rate=self.args.output_sample_rate,
            output_hop_length=self.args.output_hop_length,
            ar_mel_length_compression=self.args.gpt_code_stride_len,
            decoder_input_dim=self.args.decoder_input_dim,
            d_vector_dim=self.args.d_vector_dim,
            cond_d_vector_in_each_upsampling_layer=self.args.cond_d_vector_in_each_upsampling_layer,
        )

    # REQUIRED ABSTRACT METHOD IMPLEMENTATIONS
    def forward(self, x):
        """Required forward method implementation."""
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )

    def train_step(self, batch, criterion):
        """Required train_step method implementation."""
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )

    def eval_step(self, batch, criterion):
        """Required eval_step method implementation."""
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )

    # Perceiver LoRA Training Mode Management
    def set_perceiver_training_mode(self, mode="perceiver_lora"):
        """Set training mode specifically for Perceiver Resampler."""
        if mode == "perceiver_lora":
            self.freeze_base_perceiver_weights()
            self.train()
            self._training_mode = "perceiver_lora"
            print("🎯 Perceiver Resampler LoRA training mode activated")
        elif mode == "full":
            self.unfreeze_all_perceiver_weights()
            self.train()
            self._training_mode = "full"
            print("🔥 Full Perceiver Resampler training mode activated")
        elif mode == "inference":
            self.eval()
            self._training_mode = "inference"
            for param in self.parameters():
                param.requires_grad = False
            print("🔮 Inference mode activated")
        else:
            raise ValueError("Mode must be 'perceiver_lora', 'full', or 'inference'")

    def freeze_base_perceiver_weights(self):
        """Freeze all Perceiver Resampler parameters except LoRA adapters."""
        if self.gpt is None:
            print("⚠️ GPT model not initialized yet")
            return
            
        frozen_count = 0
        trainable_count = 0
        
        for name, param in self.gpt.named_parameters():
            if "lora_" in name and ("perceiver" in name.lower() or "resampler" in name.lower()):
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()
        
        if trainable_count > 0:
            print(f"✅ Frozen {frozen_count:,} base parameters")
            print(f"🎯 Trainable Perceiver LoRA parameters: {trainable_count:,}")
            print(f"📊 Parameter efficiency: {trainable_count/(frozen_count + trainable_count)*100:.2f}%")
        else:
            print("⚠️ No Perceiver LoRA parameters found. Check your integration.")

    def unfreeze_all_perceiver_weights(self):
        """Unfreeze all Perceiver Resampler parameters for full fine-tuning."""
        if self.gpt is None:
            print("⚠️ GPT model not initialized yet")
            return
            
        perceiver_params = 0
        for name, param in self.gpt.named_parameters():
            if "perceiver" in name.lower() or "resampler" in name.lower():
                param.requires_grad = True
                perceiver_params += param.numel()
        
        print(f"🔓 {perceiver_params:,} Perceiver Resampler parameters unfrozen for full training")

    def get_perceiver_lora_parameters(self):
        """Get only Perceiver Resampler LoRA parameters for optimizer."""
        if self.gpt is None:
            return []
        return [p for n, p in self.gpt.named_parameters() 
                if "lora_" in n and ("perceiver" in n.lower() or "resampler" in n.lower()) and p.requires_grad]

    def get_perceiver_parameters(self):
        """Get all Perceiver Resampler parameters (LoRA + base)."""
        if self.gpt is None:
            return []
        return [p for n, p in self.gpt.named_parameters() 
                if ("perceiver" in n.lower() or "resampler" in n.lower()) and p.requires_grad]

    def print_perceiver_parameter_status(self):
        """Print detailed Perceiver Resampler parameter status."""
        total_perceiver = 0
        trainable_perceiver = 0
        lora_perceiver = 0
        
        for name, param in self.gpt.named_parameters():
            if "perceiver" in name.lower() or "resampler" in name.lower():
                total_perceiver += param.numel()
                if param.requires_grad:
                    trainable_perceiver += param.numel()
                    if "lora_" in name:
                        lora_perceiver += param.numel()
        
        print(f"\n🎯 Perceiver Resampler Parameter Status:")
        print(f"├── Total Perceiver: {total_perceiver:,}")
        print(f"├── Trainable Perceiver: {trainable_perceiver:,} ({trainable_perceiver/total_perceiver*100:.2f}%)")
        print(f"├── LoRA Perceiver: {lora_perceiver:,} ({lora_perceiver/total_perceiver*100:.2f}%)")
        print(f"├── Frozen Perceiver: {total_perceiver-trainable_perceiver:,} ({(total_perceiver-trainable_perceiver)/total_perceiver*100:.2f}%)")
        print(f"└── Training Mode: {self._training_mode}")

    def save_perceiver_lora_checkpoint(self, checkpoint_path):
        """Save only Perceiver Resampler LoRA parameters."""
        perceiver_lora_state_dict = {
            k: v.cpu() for k, v in self.state_dict().items() 
            if "lora_" in k and ("perceiver" in k.lower() or "resampler" in k.lower())
        }
        
        if not perceiver_lora_state_dict:
            print("⚠️ No Perceiver LoRA parameters found to save!")
            return
        
        torch.save({
            'perceiver_lora_state_dict': perceiver_lora_state_dict,
            'perceiver_lora_config': {
                'r': getattr(self.args, 'perceiver_lora_rank', 8),
                'alpha': getattr(self.args, 'perceiver_lora_alpha', 16),
                'dropout': getattr(self.args, 'perceiver_lora_dropout', 0.05)
            },
            'training_mode': self._training_mode
        }, checkpoint_path)
        
        print(f"💾 Perceiver LoRA checkpoint saved: {checkpoint_path}")
        print(f"🎯 Saved {len(perceiver_lora_state_dict)} Perceiver LoRA parameter tensors")

    def load_perceiver_lora_checkpoint(self, checkpoint_path):
        """Load Perceiver Resampler LoRA parameters."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        perceiver_lora_state_dict = checkpoint['perceiver_lora_state_dict']
        
        missing_keys, unexpected_keys = self.load_state_dict(perceiver_lora_state_dict, strict=False)
        
        print(f"🔄 Perceiver LoRA checkpoint loaded: {checkpoint_path}")
        if missing_keys:
            print(f"⚠️ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
        
        if 'training_mode' in checkpoint:
            self.set_perceiver_training_mode(checkpoint['training_mode'])

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        """Compute the conditioning latents for the GPT model from the given audio."""
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]
        if self.args.gpt_use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i : i + 22050 * chunk_length]
                if audio_chunk.size(-1) < 22050 * 0.33:
                    continue
                mel_chunk = wav_to_mel_cloning(
                    audio_chunk,
                    mel_norms=self.mel_stats.cpu(),
                    n_fft=2048,
                    hop_length=256,
                    win_length=1024,
                    power=2,
                    normalized=False,
                    sample_rate=22050,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                )
                style_emb = self.gpt.get_style_emb(mel_chunk.to(self.device), None)
                style_embs.append(style_emb)
            cond_latent = torch.stack(style_embs).mean(dim=0)
        else:
            mel = wav_to_mel_cloning(
                audio,
                mel_norms=self.mel_stats.cpu(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
            )
            cond_latent = self.gpt.get_style_emb(mel.to(self.device))
        return cond_latent.transpose(1, 2)

    @torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (
            self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.device), l2_norm=True)
            .unsqueeze(-1)
            .to(self.device)
        )

    @torch.inference_mode()
    def get_conditioning_latents(
        self,
        audio_path,
        max_ref_length=30,
        gpt_cond_len=6,
        gpt_cond_chunk_len=6,
        librosa_trim_db=None,
        sound_norm_refs=False,
        load_sr=22050,
    ):
        """Get the conditioning latents for the GPT model from the given audio."""
        if not isinstance(audio_path, list):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path

        speaker_embeddings = []
        audios = []
        speaker_embedding = None
        for file_path in audio_paths:
            audio = load_audio(file_path, load_sr)
            audio = audio[:, : load_sr * max_ref_length].to(self.device)
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

            speaker_embedding = self.get_speaker_embedding(audio, load_sr)
            speaker_embeddings.append(speaker_embedding)
            audios.append(audio)

        full_audio = torch.cat(audios, dim=-1)
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
        )

        if speaker_embeddings:
            speaker_embedding = torch.stack(speaker_embeddings)
            speaker_embedding = speaker_embedding.mean(dim=0)

        return gpt_cond_latents, speaker_embedding

    def synthesize(self, text, config, speaker_wav, language, speaker_id=None, **kwargs):
        """Synthesize speech with the given input text."""
        assert (
            "zh-cn" if language == "zh" else language in self.config.languages
        ), f" ❗ Language {language} is not supported. Supported languages are {self.config.languages}"
        
        settings = {
            "temperature": config.temperature,
            "length_penalty": config.length_penalty,
            "repetition_penalty": config.repetition_penalty,
            "top_k": config.top_k,
            "top_p": config.top_p,
        }
        settings.update(kwargs)
        
        if speaker_id is not None:
            gpt_cond_latent, speaker_embedding = self.speaker_manager.speakers[speaker_id].values()
            return self.inference(text, language, gpt_cond_latent, speaker_embedding, **settings)
        
        settings.update({
            "gpt_cond_len": config.gpt_cond_len,
            "gpt_cond_chunk_len": config.gpt_cond_chunk_len,
            "max_ref_len": config.max_ref_len,
            "sound_norm_refs": config.sound_norm_refs,
        })
        return self.full_inference(text, speaker_wav, language, **settings)

    @torch.inference_mode()
    def full_inference(
        self,
        text,
        ref_audio_path,
        language,
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        gpt_cond_len=30,
        gpt_cond_chunk_len=6,
        max_ref_len=10,
        sound_norm_refs=False,
        **hf_generate_kwargs,
    ):
        """This function produces an audio clip of the given text being spoken with the given reference voice."""
        (gpt_cond_latent, speaker_embedding) = self.get_conditioning_latents(
            audio_path=ref_audio_path,
            gpt_cond_len=gpt_cond_len,
            gpt_cond_chunk_len=gpt_cond_chunk_len,
            max_ref_length=max_ref_len,
            sound_norm_refs=sound_norm_refs,
        )

        return self.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            **hf_generate_kwargs,
        )

    @torch.inference_mode()
    def inference(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        num_beams=1,
        speed=1.0,
        enable_text_splitting=False,
        **hf_generate_kwargs,
    ):
        language = language.split("-")[0]
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        
        if enable_text_splitting:
            text = split_sentence(text, language, self.tokenizer.char_limits.get(language, 250))
        else:
            text = [text]

        wavs = []
        gpt_latents_list = []
        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(self.device)

            assert (
                text_tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

            with torch.no_grad():
                gpt_codes = self.gpt.generate(
                    cond_latents=gpt_cond_latent,
                    text_inputs=text_tokens,
                    input_tokens=None,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=self.gpt_batch_size,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    output_attentions=False,
                    **hf_generate_kwargs,
                )

                expected_output_len = torch.tensor(
                    [gpt_codes.shape[-1] * self.gpt.code_stride_len], device=text_tokens.device
                )

                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                gpt_latents = self.gpt(
                    text_tokens,
                    text_len,
                    gpt_codes,
                    expected_output_len,
                    cond_latents=gpt_cond_latent,
                    return_attentions=False,
                    return_latent=True,
                )

                if length_scale != 1.0:
                    gpt_latents = F.interpolate(
                        gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                    ).transpose(1, 2)

                gpt_latents_list.append(gpt_latents.cpu())
                wavs.append(self.hifigan_decoder(gpt_latents, g=speaker_embedding).cpu().squeeze())

            torch.cuda.empty_cache()

        return {
            "wav": torch.cat(wavs, dim=0).numpy(),
            "gpt_latents": torch.cat(gpt_latents_list, dim=1).numpy(),
            "speaker_embedding": speaker_embedding,
        }

    def handle_chunks(self, wav_gen, wav_gen_prev, wav_overlap, overlap_len):
        """Handle chunk formatting in streaming mode"""
        wav_chunk = wav_gen[:-overlap_len]
        if wav_gen_prev is not None:
            wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) : -overlap_len]
        if wav_overlap is not None:
            if overlap_len > len(wav_chunk):
                if wav_gen_prev is not None:
                    wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) :]
                else:
                    wav_chunk = wav_gen[-overlap_len:]
                return wav_chunk, wav_gen, None
            else:
                crossfade_wav = wav_chunk[:overlap_len]
                crossfade_wav = crossfade_wav * torch.linspace(0.0, 1.0, overlap_len).to(crossfade_wav.device)
                wav_chunk[:overlap_len] = wav_overlap * torch.linspace(1.0, 0.0, overlap_len).to(wav_overlap.device)
                wav_chunk[:overlap_len] += crossfade_wav

        wav_overlap = wav_gen[-overlap_len:]
        wav_gen_prev = wav_gen
        return wav_chunk, wav_gen_prev, wav_overlap

    @torch.inference_mode()
    def inference_stream(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        stream_chunk_size=20,
        overlap_wav_len=1024,
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        speed=1.0,
        enable_text_splitting=False,
        **hf_generate_kwargs,
    ):
        language = language.split("-")[0]
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        
        if enable_text_splitting:
            text = split_sentence(text, language, self.tokenizer.char_limits.get(language, 250))
        else:
            text = [text]

        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(self.device)

            assert (
                text_tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

            fake_inputs = self.gpt.compute_embeddings(
                gpt_cond_latent.to(self.device),
                text_tokens,
            )
            gpt_generator = self.gpt.get_generator(
                fake_inputs=fake_inputs,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=1,
                num_return_sequences=1,
                length_penalty=float(length_penalty),
                repetition_penalty=float(repetition_penalty),
                output_attentions=False,
                output_hidden_states=True,
                **hf_generate_kwargs,
            )

            last_tokens = []
            all_latents = []
            wav_gen_prev = None
            wav_overlap = None
            is_end = False

            while not is_end:
                try:
                    x, latent = next(gpt_generator)
                    last_tokens += [x]
                    all_latents += [latent]
                except StopIteration:
                    is_end = True

                if is_end or (stream_chunk_size > 0 and len(last_tokens) >= stream_chunk_size):
                    gpt_latents = torch.cat(all_latents, dim=0)[None, :]
                    if length_scale != 1.0:
                        gpt_latents = F.interpolate(
                            gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                        ).transpose(1, 2)
                    wav_gen = self.hifigan_decoder(gpt_latents, g=speaker_embedding.to(self.device))
                    wav_chunk, wav_gen_prev, wav_overlap = self.handle_chunks(
                        wav_gen.squeeze(), wav_gen_prev, wav_overlap, overlap_wav_len
                    )
                    last_tokens = []
                    yield wav_chunk

    @staticmethod
    def init_from_config(config: "XttsConfig", **kwargs):
        return Xtts(config)

    def eval(self):
        """Sets the model to evaluation mode."""
        if self.gpt is not None:
            self.gpt.init_gpt_for_inference()
        super().eval()

    def get_compatible_checkpoint_state_dict(self, model_path):
        checkpoint = load_fsspec(model_path, map_location=torch.device("cpu"))["model"]
        ignore_keys = ["torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae"]
        for key in list(checkpoint.keys()):
            if key.startswith("xtts."):
                new_key = key.replace("xtts.", "")
                checkpoint[new_key] = checkpoint[key]
                del checkpoint[key]
                key = new_key

            if key.split(".")[0] in ignore_keys:
                del checkpoint[key]

        return checkpoint

    def load_checkpoint(
        self,
        config,
        checkpoint_dir=None,
        checkpoint_path=None,
        vocab_path=None,
        eval=True,
        strict=True,
        use_deepspeed=False,
        speaker_file_path=None,
    ):
        """Loads a checkpoint from disk and initializes the model's state and tokenizer."""
        model_path = checkpoint_path or os.path.join(checkpoint_dir, "model.pth")
        vocab_path = vocab_path or os.path.join(checkpoint_dir, "vocab.json")

        if speaker_file_path is None and checkpoint_dir is not None:
            speaker_file_path = os.path.join(checkpoint_dir, "speakers_xtts.pth")

        self.language_manager = LanguageManager(config)
        self.speaker_manager = None
        if speaker_file_path is not None and os.path.exists(speaker_file_path):
            self.speaker_manager = SpeakerManager(speaker_file_path)

        if os.path.exists(vocab_path):
            self.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)

        self.init_models()

        checkpoint = self.get_compatible_checkpoint_state_dict(model_path)

        # Check if model has LoRA parameters
        model_has_lora = any("lora_" in name for name, _ in self.named_parameters())
        checkpoint_has_lora = any("lora_" in key for key in checkpoint.keys())
        
        if model_has_lora and not checkpoint_has_lora:
            print("⚠️ Model has LoRA parameters but checkpoint doesn't. Using non-strict loading.")
            strict = False

        try:
            missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=strict)
            if missing_keys:
                lora_missing = [k for k in missing_keys if "lora_" in k]
                other_missing = [k for k in missing_keys if "lora_" not in k]
                
                if lora_missing:
                    print(f"📝 LoRA parameters initialized randomly: {len(lora_missing)} parameters")
                if other_missing:
                    print(f"⚠️ Other missing parameters: {other_missing}")
                    
        except Exception as e:
            if eval:
                self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache)
            missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)
            print(f"⚠️ Fallback loading completed with {len(missing_keys)} missing keys")

        if eval:
            self.hifigan_decoder.eval()
            self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=use_deepspeed)
            self.gpt.eval()