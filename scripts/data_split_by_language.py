import json
import glob
import os
import pandas as pd
from datasets import Dataset
import argparse
from data_aggregation import extract_aligned_chunks

def process_languages(data_dir: str, output_base: str):
    splits = ["train", "dev", "test"]
    target_langs = {'en': 'english', 'fr': 'french', 'de': 'german'}
    
    for split in splits:
        search_pattern = f"{data_dir}/**/*_{split}*.jsonl"
        file_paths = glob.glob(search_pattern, recursive=True)
        
        # exclude masked files and ensure we don't catch accidental splits
        filtered_paths = []
        for p in file_paths:
            filename = os.path.basename(p)
            if "masked" in filename:
                continue
            if f"_{split}_" in filename or f"_{split}-" in filename:
                filtered_paths.append(p)
        
        # Dictionary to hold data for each language
        language_buckets = {lang: [] for lang in target_langs.values()}
        
        for file_path in filtered_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    doc = json.loads(line)
                    
                    language_code = doc.get("document_metadata", {}).get("language", "unknown")
                    if language_code not in target_langs:
                        continue
                        
                    lang_name = target_langs[language_code]
                    raw_ocr = doc.get("ocr_hypothesis", {}).get("transcription_unit", "")
                    raw_gt = doc.get("ground_truth", {}).get("transcription_unit", "")
                    
                    chunks = extract_aligned_chunks(raw_ocr, raw_gt, target_words=300)
                    for chunk in chunks:
                        language_buckets[lang_name].append({
                            "dataset": doc.get("document_metadata", {}).get("primary_dataset_name", "unknown"),
                            "ocr_text": chunk["ocr_text"],
                            "ground_truth": chunk["ground_truth"]
                        })

        # Save each language bucket
        os.makedirs(os.path.join(output_base, "languages"), exist_ok=True)
        for lang_name, data in language_buckets.items():
            if not data: continue
            
            df = pd.DataFrame(data)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            output_file = os.path.join(output_base, "languages", f"{lang_name}_{split}.parquet")
            Dataset.from_pandas(df).to_parquet(output_file)
            print(f"Saved {lang_name} ({split}): {len(df)} chunks")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split HIPE data by language")
    parser.add_argument("-i","--input_path",type=str,default="./HIPE-OCRepair-2026-data/data/v0.9",help="Input data dir")
    parser.add_argument("-o","--output_path",type=str,default="./data/",help="Output directory")
    
    args = parser.parse_args()
    process_languages(args.input_path, args.output_path)
