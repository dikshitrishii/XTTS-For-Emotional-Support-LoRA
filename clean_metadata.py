import csv
import os

def clean_metadata_inplace(file_path):
    # Read and clean all lines into memory
    cleaned_rows = []
    with open(file_path, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin, delimiter="|")
        for idx, row in enumerate(reader, 1):
            if len(row) == 2:
                audio_file, text = row
                text = text.replace("|", "｜")  # Replace | with full-width vertical bar
                cleaned_rows.append([audio_file, text])
            else:
                print(f"Skipping malformed line {idx}: {row}")
    
    # Write cleaned lines back to the same file
    with open(file_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, delimiter="|")
        writer.writerows(cleaned_rows)

clean_metadata_inplace("datasets-1/metadata_train.csv")
clean_metadata_inplace("datasets-1/metadata_eval.csv")
