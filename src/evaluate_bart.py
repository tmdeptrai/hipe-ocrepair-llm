import torch
import pandas as pd
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from metrics import get_comparative_report, get_stats
from typing import List, Dict

def chunk_inference(text: str, model, tokenizer, device, chunk_size=256, overlap=32):
    """
    Inference with sliding window for long text chunks.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    
    if len(input_ids) <= chunk_size:
        with torch.no_grad():
            output = model.generate(input_ids.unsqueeze(0).to(device), max_new_tokens=chunk_size+64)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    all_outputs = []
    for i in range(0, len(input_ids), chunk_size - overlap):
        chunk = input_ids[i:i + chunk_size].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.generate(chunk, max_new_tokens=chunk_size + 64)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        all_outputs.append(decoded.strip())
    
    return " ".join(all_outputs)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned BART on HIPE chunks.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned BART model/adapter")
    parser.add_argument("--data_path", type=str, default="data/hipe_aggregated_test.parquet", help="Path to test parquet")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1, help="Currently supports 1 for sliding window logic")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for quick testing")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(args.device)
    
    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    
    if args.limit:
        df = df.head(args.limit)
    
    results = []
    
    print(f"Running inference on {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        ocr_text = row['ocr_text']
        gt_text = row['ground_truth']
        
        corrected_text = chunk_inference(ocr_text, model, tokenizer, args.device)
        
        results.append({
            "document_id": row.get('document_id', 'unknown'),
            "chunk_idx": row.get('chunk_idx', 0),
            "language": row['language'],
            "ocr_text": ocr_text,
            "ground_truth": gt_text,
            "model_output": corrected_text
        })
    
    res_df = pd.DataFrame(results)
    
    # Global Metrics
    print("\n" + "="*50)
    print("GLOBAL EVALUATION REPORT")
    print("="*50)
    
    report = get_comparative_report(
        res_df['ground_truth'].tolist(),
        res_df['ocr_text'].tolist(),
        res_df['model_output'].tolist()
    )
    
    print(f"Original CER:  {report['ocr_stats']['mean']:.4f} ± {report['ocr_stats']['std']:.4f}")
    print(f"Model CER:     {report['model_stats']['mean']:.4f} ± {report['model_stats']['std']:.4f}")
    print(f"CER Reduction: {report['relative_cer_reduction']*100:.2f}%")
    print(f"Cohen's d:     {report['cohens_d']:.4f}")
    
    # Per-Language Breakdown
    print("\nPER-LANGUAGE BREAKDOWN (Cohen's d)")
    for lang in res_df['language'].unique():
        lang_df = res_df[res_df['language'] == lang]
        lang_report = get_comparative_report(
            lang_df['ground_truth'].tolist(),
            lang_df['ocr_text'].tolist(),
            lang_df['model_output'].tolist()
        )
        print(f"- {lang:<3}: d={lang_report['cohens_d']:.4f} | RER={lang_report['relative_cer_reduction']*100:.1f}%")

    output_file = "evaluation_results.parquet"
    res_df.to_parquet(output_file)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
