import json
import glob
import pandas as pd
from datasets import Dataset
import argparse
import difflib

def extract_aligned_chunks(ocr_text, gt_text, target_words=300):
    """
    Ensure the splits happen at exact matching anchor points.
    """
    ocr_words = str(ocr_text).split()
    gt_words = str(gt_text).split()
    
    if len(ocr_words) < target_words and len(gt_words) < target_words:
        return [{"ocr_text": str(ocr_text), "ground_truth": str(gt_text)}]
        
    matcher = difflib.SequenceMatcher(None, ocr_words, gt_words)
    matching_blocks = matcher.get_matching_blocks()
    
    chunks = []
    last_ocr_idx = 0
    last_gt_idx = 0
    
    for block in matching_blocks:
        ocr_start, gt_start, match_length = block
        if match_length == 0: break
            
        current_chunk_size = ocr_start - last_ocr_idx
        
        if current_chunk_size >= target_words:
            chunk_ocr = " ".join(ocr_words[last_ocr_idx:ocr_start])
            chunk_gt = " ".join(gt_words[last_gt_idx:gt_start])
            
            if chunk_ocr.strip() or chunk_gt.strip():
                chunks.append({"ocr_text": chunk_ocr, "ground_truth": chunk_gt})
            
            last_ocr_idx = ocr_start
            last_gt_idx = gt_start
            
    # catch the remaining tail
    final_ocr = " ".join(ocr_words[last_ocr_idx:])
    final_gt = " ".join(gt_words[last_gt_idx:])
    if final_ocr.strip() or final_gt.strip():
        chunks.append({"ocr_text": final_ocr, "ground_truth": final_gt})
        
    return chunks

def aggregate_split(data_dir: str, split_name: str) -> Dataset:
    search_pattern = f"{data_dir}/**/*_{split_name}_*.jsonl"
    file_paths = glob.glob(search_pattern, recursive=True)
    
    extracted_data = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                doc = json.loads(line)
                
                dataset_name = doc.get("document_metadata", {}).get("primary_dataset_name", "unknown")
                language = doc.get("document_metadata", {}).get("language", "unknown")
                raw_ocr = doc.get("ocr_hypothesis", {}).get("transcription_unit", "")
                raw_gt = doc.get("ground_truth", {}).get("transcription_unit", "")
                
                # Apply the aligned chunking
                chunks = extract_aligned_chunks(raw_ocr, raw_gt, target_words=300)
                
                for chunk in chunks:
                    extracted_data.append({
                        "dataset": dataset_name,
                        "language": language,
                        "ocr_text": chunk["ocr_text"],
                        "ground_truth": chunk["ground_truth"]
                    })
                
    df = pd.DataFrame(extracted_data)
    
    if not df.empty:
        print(f"\n--- {split_name.upper()} SPLIT AGGREGATED ---")
        print(f"Total Chunks Generated: {len(df)}")
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

    train_dataset.to_parquet(args.output_path + "hipe_aggregated_train.parquet")
    dev_dataset.to_parquet(args.output_path + "hipe_aggregated_dev.parquet")
    
    print("\nFiles chunked and saved! Ready for fast training.")