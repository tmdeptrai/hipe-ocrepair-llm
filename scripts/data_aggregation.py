import json
import glob
import os
import pandas as pd
from datasets import Dataset
import argparse
import difflib

def extract_aligned_chunks(ocr_text, gt_text, target_words=300, window_size=100):
    """
    Ensure the splits happen at exact matching anchor points, prioritizing 
    sentence boundaries (punctuation) within a flexible window.
    """
    ocr_words = str(ocr_text).split()
    gt_words = str(gt_text).split()
    
    if len(ocr_words) < target_words and len(gt_words) < target_words:
        return [{"ocr_text": str(ocr_text), "ground_truth": str(gt_text)}]
        
    matcher = difflib.SequenceMatcher(None, ocr_words, gt_words)
    matching_blocks = matcher.get_matching_blocks()
    
    # All possible safe split points (anchors)
    # A split is safe at any word boundary where OCR and GT are identical
    anchors = []
    for block in matching_blocks:
        ocr_start, gt_start, match_length = block
        if match_length == 0: continue
        for offset in range(match_length + 1):
            anchors.append((ocr_start + offset, gt_start + offset))
            
    # Sort and unique anchors
    anchors = sorted(list(set(anchors)))
    
    chunks = []
    last_ocr_idx = 0
    last_gt_idx = 0
    
    i = 0
    while i < len(anchors):
        ocr_idx, gt_idx = anchors[i]
        current_chunk_size = ocr_idx - last_ocr_idx
        
        if current_chunk_size >= target_words:
            # Look ahead for a better split point (punctuation) within window
            best_anchor_idx = i
            
            for j in range(i, len(anchors)):
                l_ocr, l_gt = anchors[j]
                if (l_ocr - last_ocr_idx) > (target_words + window_size):
                    break
                
                # Check if word before this anchor ends in punctuation
                if l_ocr > 0:
                    prev_word = ocr_words[l_ocr - 1]
                    # Common sentence/clause terminators (including historical ones)
                    if prev_word.endswith(('.', '!', '?', ';', ':')):
                        best_anchor_idx = j
                        break
            
            s_ocr, s_gt = anchors[best_anchor_idx]
            
            chunk_ocr = " ".join(ocr_words[last_ocr_idx:s_ocr])
            chunk_gt = " ".join(gt_words[last_gt_idx:s_gt])
            
            if chunk_ocr.strip() or chunk_gt.strip():
                chunks.append({"ocr_text": chunk_ocr, "ground_truth": chunk_gt})
                
            last_ocr_idx, last_gt_idx = s_ocr, s_gt
            # Catch up index i to the new split point
            while i < len(anchors) and anchors[i][0] < last_ocr_idx:
                i += 1
            continue
            
        i += 1
            
    # catch the remaining tail
    final_ocr = " ".join(ocr_words[last_ocr_idx:])
    final_gt = " ".join(gt_words[last_gt_idx:])
    if final_ocr.strip() or final_gt.strip():
        chunks.append({"ocr_text": final_ocr, "ground_truth": final_gt})
        
    return chunks

def aggregate_split(data_dir: str, split_name: str) -> Dataset:
    search_pattern = f"{data_dir}/**/*_{split_name}*.jsonl"
    file_paths = glob.glob(search_pattern, recursive=True)
    
    # exclude masked files and ensure we don't catch accidental splits
    # e.g. for 'dev', don't want to catch 'dev-unmatched' if we only want 'dev'
    # BUT for dta19, 'test-unmatched' IS the target.
    # So we check if the filename contains split_name followed by either '_' or '-'
    filtered_paths = []
    for p in file_paths:
        filename = os.path.basename(p)
        if "masked" in filename:
            continue
        # Check for _test_ or _test-
        if f"_{split_name}_" in filename or f"_{split_name}-" in filename:
            filtered_paths.append(p)
    
    extracted_data = []
    
    for file_path in filtered_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                doc = json.loads(line)
                
                dataset_name = doc.get("document_metadata", {}).get("primary_dataset_name", "unknown")
                language = doc.get("document_metadata", {}).get("language", "unknown")
                doc_id = doc.get("document_metadata", {}).get("document_id", "unknown")
                raw_ocr = doc.get("ocr_hypothesis", {}).get("transcription_unit", "")
                raw_gt = doc.get("ground_truth", {}).get("transcription_unit", "")
                
                # Apply the aligned chunking
                chunks = extract_aligned_chunks(raw_ocr, raw_gt, target_words=300)
                
                for idx, chunk in enumerate(chunks):
                    extracted_data.append({
                        "document_id": doc_id,
                        "chunk_idx": idx,
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
    
    # Only shuffle training data
    if split_name == "train":
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
    return Dataset.from_pandas(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate all datasets to prepare for fine tuning")
    parser.add_argument("-i","--input_path",type=str,default="./HIPE-OCRepair-2026-data/data/v0.9",help="Directory containing all the datasets in .jsonl format")
    parser.add_argument("-o","--output_path",type=str,default="./data/",help="Name of the output files for train/test in parquet format")
    
    args = parser.parse_args()
    base_dir = args.input_path

    train_dataset = aggregate_split(base_dir, "train")
    dev_dataset = aggregate_split(base_dir, "dev")
    test_dataset = aggregate_split(base_dir, "test")

    train_dataset.to_parquet(args.output_path + "hipe_aggregated_train.parquet")
    dev_dataset.to_parquet(args.output_path + "hipe_aggregated_dev.parquet")
    test_dataset.to_parquet(args.output_path + "hipe_aggregated_test.parquet")
    
    print("\nFiles chunked and saved! Ready for training.")