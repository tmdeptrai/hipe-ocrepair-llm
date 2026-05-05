import pandas as pd
import json
import argparse
import os

def get_text_stats(text):
    """Helper to ensure required schema fields are generated."""
    text = str(text) if pd.notnull(text) else ""
    return {
        "transcription_unit": text,
        "num_chars": len(text),
        "num_tokens": len(text.split())
    }

def write_jsonl_pair(df_subset, ref_dir, hyp_dir, team_name, run_name, dataset_tag, lang_tag):
    """Writes a matched pair of Reference and Hypothesis JSONL files."""
    # Following official naming conventions: hipe-ocrepair-bench_v0.9_<dataset>_<primary_version>_<split>_<language>.jsonl
    base_filename = f"hipe-ocrepair-bench_v0.9_{dataset_tag}_v1.0_test_{lang_tag}.jsonl"
    
    ref_filename = os.path.join(ref_dir, base_filename)
    hyp_filename = os.path.join(hyp_dir, f"{team_name}_{base_filename.replace('.jsonl', f'_{run_name}.jsonl')}")
    
    ref_records = []
    hyp_records = []
    
    for _, row in df_subset.iterrows():
        doc_id = row.get("document_id", "unknown_id")
        if "chunk_idx" in row and row["chunk_idx"] > 0:
            doc_id = f"{doc_id}_chunk{row['chunk_idx']}"
            
        dataset = row.get("dataset", "unknown")
        
        doc_metadata = {
            "document_id": doc_id,
            "primary_dataset_name": dataset
        }
        
        # 1. Build Reference Record
        ref_records.append({
            "document_metadata": doc_metadata,
            "ground_truth": get_text_stats(row["ground_truth"]),
            "ocr_hypothesis": get_text_stats(row["ocr_text"])
        })
        
        # 2. Build Hypothesis (Model Output) Record
        hyp_records.append({
            "document_metadata": doc_metadata,
            "ocr_postcorrection_output": get_text_stats(row["model_output"])
        })
        
    # Write Reference JSONL
    with open(ref_filename, 'w', encoding='utf-8') as f:
        for record in ref_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    # Write Hypothesis JSONL
    with open(hyp_filename, 'w', encoding='utf-8') as f:
        for record in hyp_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"Generated split: {dataset_tag} / {lang_tag} (Samples: {len(df_subset)})")

def split_and_create_jsonl(df, output_dir, run_name="run1", team_name="myteam"):
    ref_dir = os.path.join(output_dir, "reference")
    hyp_dir = os.path.join(output_dir, "hypothesis")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(hyp_dir, exist_ok=True)
    
    print(f"Saving splits to {output_dir}/reference and {output_dir}/hypothesis...\n")
    
    # 1. Generate the 5 Dataset Splits
    if 'dataset' in df.columns:
        for dataset in df['dataset'].dropna().unique():
            subset = df[df['dataset'] == dataset]
            write_jsonl_pair(subset, ref_dir, hyp_dir, team_name, run_name, dataset_tag=dataset, lang_tag="all")
            
    # 2. Generate the 3 Language Splits
    if 'language' in df.columns:
        for lang in df['language'].dropna().unique():
            subset = df[df['language'] == lang]
            write_jsonl_pair(subset, ref_dir, hyp_dir, team_name, run_name, dataset_tag="all", lang_tag=lang)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Parquet evaluation logs into 8 HIPE JSONL subsets.")
    parser.add_argument("--input", type=str, required=True, help="Path to the evaluated .parquet file")
    parser.add_argument("--output_dir", type=str, default="official_scorer_data", help="Directory to save JSONL splits")
    parser.add_argument("--team", type=str, default="UnivLR", help="Your team name for the file prefix")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    split_and_create_jsonl(df, args.output_dir, team_name=args.team)