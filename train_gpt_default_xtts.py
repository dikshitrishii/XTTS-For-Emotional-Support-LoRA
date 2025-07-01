import os
import gc
import pandas as pd
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
import argparse

def create_xtts_trainer_parser():
    parser = argparse.ArgumentParser(description="Arguments for XTTS Trainer")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to pretrained + checkpoint model")
    parser.add_argument("--metadatas", nargs='+', type=str, required=True,
                        help="train_csv_path,eval_csv_path,language")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Mini batch size")
    parser.add_argument("--grad_acumm", type=int, default=1,
                        help="Grad accumulation steps")
    parser.add_argument("--max_audio_length", type=int, default=255995,
                        help="Max audio length")
    parser.add_argument("--max_text_length", type=int, default=200,
                        help="Max text length")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--save_step", type=int, default=5000,
                        help="Save step")
    return parser

def train_gpt(metadatas, num_epochs, batch_size, grad_acumm, output_path, max_audio_length, max_text_length, lr, weight_decay, save_step):
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None
    OUT_PATH = output_path
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
    START_WITH_EVAL = False
    BATCH_SIZE = batch_size
    GRAD_ACUMM_STEPS = grad_acumm

    DATASETS_CONFIG_LIST = []
    for metadata in metadatas:
        train_csv, eval_csv = metadata.split(",")
        language = ''
        config_dataset = BaseDatasetConfig(
            formatter="coqui",
            dataset_name="ft_dataset",
            path=os.path.dirname(train_csv),
            meta_file_train=os.path.basename(train_csv),
            meta_file_val=os.path.basename(eval_csv),
            language=language,
        )
        DATASETS_CONFIG_LIST.append(config_dataset)

    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    DVAE_CHECKPOINT_LINK = f"https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print(" > Downloading DVAE files!")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

    DVAE_CHECKPOINT = DVAE_CHECKPOINT.replace(".pth", f"_{language}.pth")
    if not os.path.isfile(DVAE_CHECKPOINT):
        print(f" > DVAE checkpoint for language {language} not found, using the original one.")
        DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "dvae.pth")
    else:
        print(f" > Using DVAE checkpoint for language {language}: {DVAE_CHECKPOINT}")

    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    XTTS_CONFIG_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"

    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))

    if not os.path.isfile(TOKENIZER_FILE):
        print(" > Downloading XTTS v2.0 tokenizer!")
        ModelManager._download_model_files([TOKENIZER_FILE_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)
    if not os.path.isfile(XTTS_CHECKPOINT):
        print(" > Downloading XTTS v2.0 checkpoint!")
        ModelManager._download_model_files([XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)
    if not os.path.isfile(XTTS_CONFIG_FILE):
        print(" > Downloading XTTS v2.0 config!")
        ModelManager._download_model_files([XTTS_CONFIG_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=11025,
        debug_loading_failures=False,
        max_wav_length=max_audio_length,
        max_text_length=max_text_length,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    config = GPTTrainerConfig()
    config.load_json(XTTS_CONFIG_FILE)
    config.epochs = num_epochs
    config.output_path = OUT_PATH
    config.model_args = model_args
    config.run_name = RUN_NAME
    config.project_name = PROJECT_NAME
    config.run_description = "GPT XTTS training"
    config.dashboard_logger = DASHBOARD_LOGGER
    config.logger_uri = LOGGER_URI
    config.audio = audio_config
    config.batch_size = BATCH_SIZE
    config.num_loader_workers = 8
    config.eval_split_max_size = 256
    config.print_step = 50
    config.plot_step = 100
    config.log_model_step = 100
    config.save_step = save_step
    config.save_n_checkpoints = 5
    config.save_checkpoints = True
    config.print_eval = True
    config.optimizer = "AdamW"
    config.optimizer_wd_only_on_weights = OPTIMIZER_WD_ONLY_ON_WEIGHTS
    config.optimizer_params = {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": weight_decay}
    config.lr = lr
    config.lr_scheduler = "MultiStepLR"
    config.lr_scheduler_params = {"milestones": [5000, 150000, 300000], "gamma": 0.5, "last_epoch": -1}
    config.test_sentences = []

    model = GPTTrainer.init_from_config(config)

    # --- BEGIN: Attach neutral reference audio ---
    # Load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Attach neutral reference audio path to each train/eval sample
    default_df = pd.read_csv('/home/ubuntu/Dikshit/Training/XTTSv2-Finetuning-for-Emotional-Tokens-gpt/datasets-1/default_dataset.csv', delimiter='|')
    for sample in train_samples:
        text = sample["text"]
        match = default_df[default_df["text"] == text]
        if not match.empty:
            sample["ref_file"] = match.iloc[0]["audio_file"]
        else:
            sample["ref_file"] = sample["audio_file"]
    for sample in eval_samples:
        text = sample["text"]
        match = default_df[default_df["text"] == text]
        if not match.empty:
            sample["ref_file"] = match.iloc[0]["audio_file"]
        else:
            sample["ref_file"] = sample["audio_file"]
    # --- END: Attach neutral reference audio ---

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS
        ),
        config,
        output_path=os.path.join(output_path, "run", "training"),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
    trainer_out_path = trainer.output_path

    del model, trainer, train_samples, eval_samples
    gc.collect()
    return trainer_out_path

if __name__ == "__main__":
    parser = create_xtts_trainer_parser()
    args = parser.parse_args()
    trainer_out_path = train_gpt(
        metadatas=args.metadatas,
        output_path=args.output_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_acumm=args.grad_acumm,
        weight_decay=args.weight_decay,
        lr=args.lr,
        max_text_length=args.max_text_length,
        max_audio_length=args.max_audio_length,
        save_step=args.save_step
    )
    print(f"Checkpoint saved in dir: {trainer_out_path}")
