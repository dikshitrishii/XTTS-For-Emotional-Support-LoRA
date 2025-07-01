from datasets import load_dataset
import dotenv
import os
import soundfile as sf
from tqdm.auto import tqdm
import time
import re
import math

# Load environment variables
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

download_dir = "dataset/"

# Load the Expresso dataset
ds = load_dataset(
    "ylacombe/expresso",
    token=HF_TOKEN,
    cache_dir=download_dir
)

base_dir = "datasets-2emos"
wavs_dir = os.path.join(base_dir, "wavs")
os.makedirs(wavs_dir, exist_ok=True)

def wrap_sentences_with_style(text, style):
    sentences = re.findall(r'[^.!?]+[.!?]?', text, flags=re.UNICODE)
    sentences = [s.strip() for s in sentences if s.strip()]
    wrapped = [f"<{style}>{s}</{style}>" for s in sentences]
    return " ".join(wrapped)

def save_dataset(ds, base_dir, wavs_dir, eval_fraction=0.15):
    train_metadata = []
    eval_metadata = []
    total_errors = 0
    start_time = time.time()

    # --- Filter for only happy and whisper styles ---
    desired_styles = {"happy", "whisper"}
    all_examples = [
        example for example in ds["train"]
        if example.get("style", "").lower() in desired_styles
    ]

    n_total = len(all_examples)
    n_eval = math.ceil(n_total * eval_fraction)
    n_train = n_total - n_eval

    print(f"Splitting {n_total} samples: {n_train} for train, {n_eval} for eval.")

    train_examples = all_examples[:n_train]
    eval_examples = all_examples[n_train:]

    # --- Process train samples ---
    for i, example in enumerate(train_examples):
        try:
            if isinstance(example['audio'], dict) and 'array' in example['audio']:
                audio_array = example['audio']['array']
                sampling_rate = example['audio']['sampling_rate']
            else:
                file_obj = example['audio']
                audio_array, sampling_rate = sf.read(file_obj)

            file_name = f"train_{i}.wav"
            file_path = os.path.join(wavs_dir, file_name)
            sf.write(file_path, audio_array, sampling_rate)
            relative_path = f"wavs/{file_name}"

            style = example.get("style", "unknown")
            text = example["text"]
            text_with_style = wrap_sentences_with_style(text, style)

            entry = f"{relative_path}|{text_with_style}"
            train_metadata.append(entry)
        except Exception as e:
            total_errors += 1
            continue

    # --- Process eval samples ---
    for i, example in enumerate(eval_examples):
        try:
            if isinstance(example['audio'], dict) and 'array' in example['audio']:
                audio_array = example['audio']['array']
                sampling_rate = example['audio']['sampling_rate']
            else:
                file_obj = example['audio']
                audio_array, sampling_rate = sf.read(file_obj)

            file_name = f"eval_{i}.wav"
            file_path = os.path.join(wavs_dir, file_name)
            sf.write(file_path, audio_array, sampling_rate)
            relative_path = f"wavs/{file_name}"

            style = example.get("style", "unknown")
            text = example["text"]
            text_with_style = wrap_sentences_with_style(text, style)

            entry = f"{relative_path}|{text_with_style}"
            eval_metadata.append(entry)
        except Exception as e:
            total_errors += 1
            continue

    print(f"Added {len(eval_metadata)} samples to eval set.")

    # --- Save metadata files ---
    with open(os.path.join(base_dir, "metadata_train.csv"), "w", encoding="utf-8") as f:
        f.write("audio_file|text\n")
        f.write("\n".join(train_metadata))
    print(f"- Saved metadata_train.csv with {len(train_metadata)} entries")

    with open(os.path.join(base_dir, "metadata_eval.csv"), "w", encoding="utf-8") as f:
        f.write("audio_file|text\n")
        f.write("\n".join(eval_metadata))
    print(f"- Saved metadata_eval.csv with {len(eval_metadata)} entries")

    elapsed = time.time() - start_time
    print(f"\n[SUMMARY]")
    print(f"Total processing time: {int(elapsed//60)} minutes {int(elapsed%60)} seconds")
    print(f"Files successfully processed: {len(train_metadata) + len(eval_metadata)}")
    print(f"Errors encountered: {total_errors}")
    print(f"All data saved to {base_dir} directory")

# Call the function with 15% eval split
save_dataset(ds, base_dir, wavs_dir, eval_fraction=0.15)
