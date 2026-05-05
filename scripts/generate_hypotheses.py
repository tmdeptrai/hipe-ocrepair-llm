import pandas as pd
import json
import argparse
import os
import glob

def get_text_stats(text):
    text = str(text) if pd.notnull(text) else ""
    return {
        "transcription_unit": text,
        "num_chars": len(text),
        "num_tokens": len(text.split())
    }

def generate_hypotheses_from_refs(df, ref_dir, hyp_dir, team_name="TMDUONG", run_name="run1"):
    os.makedirs(hyp_dir, exist_ok=True)
    
    print("Reconstructing full documents from chunks...")
    if 'chunk_idx' in df.columns:
        df = df.sort_values(by=['document_id', 'chunk_idx'])
        predictions = df.groupby('document_id')['model_output'].apply(
            lambda x: ' '.join(x.dropna().astype(str))
        ).to_dict()
    else:
        predictions = df.set_index('document_id')['model_output'].to_dict()
        
    search_pattern = os.path.join(ref_dir, "**", "*test*.jsonl")
    ref_files = glob.glob(search_pattern, recursive=True)
    ref_files = [f for f in ref_files if "masked" not in f and "dev" not in f and "train" not in f]
    
    print(f"Found {len(ref_files)} reference sets. Generating hypotheses...\n")
    
    for ref_path in ref_files:
        ref_base = os.path.basename(ref_path).replace('.jsonl', '')
        
        # Keep 'test' as is to match the reference filename exactly
        hyp_base_name = f"{team_name}_{ref_base}_{run_name}.jsonl"
        hyp_path = os.path.join(hyp_dir, hyp_base_name)
        
        hyp_records = []
        match_count = 0
        
        with open(ref_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                ref_record = json.loads(line)
                doc_id = ref_record["document_metadata"]["document_id"]
                
                model_output = predictions.get(doc_id, ref_record.get("ocr_hypothesis", {}).get("transcription_unit", ""))
                if doc_id in predictions: match_count += 1
                    
                hyp_records.append({
                    "document_metadata": ref_record["document_metadata"],
                    "ocr_hypothesis": ref_record["ocr_hypothesis"],
                    "ocr_postcorrection_output": get_text_stats(model_output)
                })
                
        with open(hyp_path, 'w', encoding='utf-8') as f:
            for record in hyp_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
        print(f"Generated: {hyp_base_name} ({match_count} matches)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--ref_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="official_scorer_data")
    parser.add_argument("--team", type=str, default="TMDUONG")
    parser.add_argument("--run", type=str, default="run1")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    hyp_dir = os.path.join(args.output_dir, "hypothesis")
    generate_hypotheses_from_refs(df, args.ref_dir, hyp_dir, team_name=args.team, run_name=args.run)