import argparse
from tokenizers import Tokenizer
import os
import pandas as pd
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import json
import re

def extract_styles_from_metadata(metadata_path):
    # Read the metadata file (audio_file|text)
    df = pd.read_csv(metadata_path, sep="|")
    styles = set()
    # Regex to find <style> and </style> tags in the text
    for text in df.text:
        found = re.findall(r"<(/?)(\w+)>", text)
        for slash, style in found:
            styles.add(style)
    return sorted(styles)

def combine_tokenizers(old_tokenizer, new_tokenizer, save_dir):
    json1 = json.load(open(os.path.join(old_tokenizer, 'vocab.json')))
    json2 = json.load(open(os.path.join(new_tokenizer, 'vocab.json')))

    new_vocab = {}
    idx = 0
    for word in json1.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    for word in json2.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'vocab.json'), 'w') as fp:
        json.dump(new_vocab, fp, ensure_ascii=False)

    os.system('cat {} > {}'.format(os.path.join(old_tokenizer, 'merges.txt'), os.path.join(save_dir, 'merges.txt')))
    os.system('tail -n +2 -q {} >> {}'.format(os.path.join(new_tokenizer, 'merges.txt'), os.path.join(save_dir, 'merges.txt')))

def extend_tokenizer(args):
    root = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/")
    existing_tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    old_tokenizer_path = os.path.join(root, "old_tokenizer/")
    os.makedirs(old_tokenizer_path, exist_ok=True)
    existing_tokenizer.model.save(old_tokenizer_path)

    # Extract all unique style tokens from metadata
    styles = extract_styles_from_metadata(args.metadata_path)
    special_tokens = []
    for style in styles:
        special_tokens.append(f"<{style}>")
        special_tokens.append(f"</{style}>")

    # Optionally, add your language token as well
    special_tokens.append(f"[{args.language}]")

    # Prepare training data
    traindf = pd.read_csv(args.metadata_path, sep="|")
    texts = traindf.text.to_list()

    new_tokenizer = Tokenizer(BPE())
    new_tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=args.extended_vocab_size)
    new_tokenizer.train_from_iterator(iter(texts), trainer=trainer)
    new_tokenizer.add_special_tokens(special_tokens)

    new_tokenizer_path = os.path.join(root, "new_tokenizer/")
    os.makedirs(new_tokenizer_path, exist_ok=True)
    new_tokenizer.model.save(new_tokenizer_path)

    merged_tokenizer_path = os.path.join(root, "merged_tokenizer/")
    combine_tokenizers(
        old_tokenizer_path,
        new_tokenizer_path,
        merged_tokenizer_path
    )

    # Load and update the merged tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    tokenizer.model = tokenizer.model.from_file(os.path.join(merged_tokenizer_path, 'vocab.json'), os.path.join(merged_tokenizer_path, 'merges.txt'))
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.save(os.path.join(root, "vocab.json"))

    os.system(f'rm -rf {old_tokenizer_path} {new_tokenizer_path} {merged_tokenizer_path}')

def adjust_config(args):
    config_path = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    if args.language not in config["languages"]:
        config["languages"].append(args.language)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="")
    parser.add_argument("--metadata_path", type=str, required=True, help="")
    parser.add_argument("--language", type=str, required=True, help="")
    parser.add_argument("--extended_vocab_size", default=2000, type=int, required=True, help="")
    args = parser.parse_args()

    extend_tokenizer(args)
    adjust_config(args)
