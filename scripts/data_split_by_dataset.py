import json
import glob
import os
import pandas as pd
from datasets import Dataset
import argparse
from data_aggregation import extract_aligned_chunks

def process_datasets(data_dir: str, output_base: str):
    splits = ["train", "dev", "test"]
    
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
        
        # Dictionary to hold data for each dataset
        dataset_buckets = {}
        
        for file_path in filtered_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    doc = json.loads(line)
                    
                    # Extract metadata
                    dataset_name = doc.get("document_metadata", {}).get("primary_dataset_name", "unknown")
                    language = doc.get("document_metadata", {}).get("language", "unknown")
                    raw_ocr = doc.get("ocr_hypothesis", {}).get("transcription_unit", "")
                    raw_gt = doc.get("ground_truth", {}).get("transcription_unit", "")
                    
                    if dataset_name not in dataset_buckets:
                        dataset_buckets[dataset_name] = []
                        
                    chunks = extract_aligned_chunks(raw_ocr, raw_gt, target_words=300)
                    for chunk in chunks:
                        dataset_buckets[dataset_name].append({
                            "language": language,
                            "ocr_text": chunk["ocr_text"],
                            "ground_truth": chunk["ground_truth"]
                        })

        # Save each dataset bucket
        for dataset_name, data in dataset_buckets.items():
            if not data: continue
            
            os.makedirs(os.path.join(output_base, "datasets"), exist_ok=True)
            df = pd.DataFrame(data)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            output_file = os.path.join(output_base, "datasets", f"{dataset_name}_{split}.parquet")
            Dataset.from_pandas(df).to_parquet(output_file)
            print(f"Saved {dataset_name} ({split}): {len(df)} chunks")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split HIPE data by sub-dataset")
    parser.add_argument("-i","--input_path",type=str,default="./HIPE-OCRepair-2026-data/data/v0.9",help="Input data dir")
    parser.add_argument("-o","--output_path",type=str,default="./data/",help="Output directory")
    
    args = parser.parse_args()
    process_datasets(args.input_path, args.output_path)
