import json
import glob
import pandas as pd
from datasets import Dataset
import argparse

def aggregate_split(data_dir: str, split_name: str) -> Dataset:
    """
    Finds all JSONL files for a specific split and aggregates them 
    into a single flat Dataset without dropping any data.
    """
    search_pattern = f"{data_dir}/**/*_{split_name}_*.jsonl"
    file_paths = glob.glob(search_pattern, recursive=True)
    
    extracted_data = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                doc = json.loads(line)
                
                extracted_data.append({
                    "dataset": doc.get("document_metadata", {}).get("primary_dataset_name", "unknown"),
                    "language": doc.get("document_metadata", {}).get("language", "unknown"),
                    "ocr_text": doc.get("ocr_hypothesis", {}).get("transcription_unit", ""),
                    "ground_truth": doc.get("ground_truth", {}).get("transcription_unit", "")
                })
                
    df = pd.DataFrame(extracted_data)
    
    if not df.empty:
        print(f"\n--- {split_name.upper()} SPLIT AGGREGATED ---")
        print(f"Total Rows: {len(df)}")
        print("Natural Language Distribution:")
        print(df['language'].value_counts())
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return Dataset.from_pandas(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate all datasets to prepare for fine tuning")
    parser.add_argument("-i","--input_path",type=str,default="./HIPE-OCRepair-2026-data/data/v0.9",help="Directory containing all the datasets in .jsonl format")
    parser.add_argument("-o","--output_path",type=str,default="./finetune/",help="Name of the output files for train/test in parquet format")
    
    args = parser.parse_args()
    
    base_dir = args.input_path

    train_dataset = aggregate_split(base_dir, "train")
    dev_dataset = aggregate_split(base_dir, "dev")

    train_output_path = args.output_path + "hipe_aggregated_train.parquet"
    dev_output_path = args.output_path + "hipe_aggregated_dev.parquet"
    
    train_dataset.to_parquet(train_output_path)
    dev_dataset.to_parquet(dev_output_path)
    
    print("\nFiles saved! Ready for training.")