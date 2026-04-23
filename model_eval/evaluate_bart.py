import torch
import pandas as pd
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from metrics import get_comparative_report

def batch_inference(df: pd.DataFrame, model, tokenizer, device, batch_size=16, with_metadata=False):
    """
    Performs inference in batches for massive speedups on GPU.
    Injects metadata prefix before generation and strips it afterward.
    """
    all_outputs = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Generating Corrections"):
        batch_df = df.iloc[i:i + batch_size]
        
        prompts = []
        prefixes = []
        
        for _, row in batch_df.iterrows():
            ocr_text = row['ocr_text']
            prefix = ""
            
            if with_metadata:
                year = row.get('year', None)
                dataset = row.get('dataset', 'unknown')
                lang = row.get('language', 'unknown')
                
                prefix = f"[Dataset: {dataset}, Language: {lang}"
                if year and pd.notnull(year):
                    prefix += f", Year: {int(year)}"
                prefix += "] "
            
            prefixes.append(prefix)
            prompts.append(prefix + ocr_text)
        
        # Tokenize with truncation and padding
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1024,
                num_beams=3 # Adding a small beam search improves OCR repair quality
            )
            
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Remove the prefix from the final output
        for output_text, prefix in zip(decoded, prefixes):
            output_text = output_text.strip()
            # If the model regurgitated our prefix, cleanly slice it off
            if prefix and output_text.startswith(prefix):
                output_text = output_text[len(prefix):].strip()
            all_outputs.append(output_text)
        
    return all_outputs

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned BART on HIPE chunks.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned BART model")
    parser.add_argument("--data_path", type=str, default="data/hipe_aggregated_test.parquet", help="Path to test parquet")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for quick testing")
    parser.add_argument("--with_metadata", action="store_true", help="Include metadata prefix in evaluation prompts")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(args.device)
    
    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    
    if args.limit:
        df = df.head(args.limit)
    
    print(f"Running batch inference on {len(df)} samples...")
    corrected_texts = batch_inference(
        df, 
        model, 
        tokenizer, 
        args.device, 
        batch_size=args.batch_size,
        with_metadata=args.with_metadata
    )
    
    # Attach the predictions back to the dataframe
    df['model_output'] = corrected_texts
    
    # Global Metrics
    print("\n" + "="*50)
    print("GLOBAL EVALUATION REPORT")
    print("="*50)
    
    report = get_comparative_report(
        df['ground_truth'].tolist(),
        df['ocr_text'].tolist(),
        df['model_output'].tolist()
    )
    
    print("--- Character Error Rate (CER) ---")
    print(f"Original CER:  {report['ocr_cer_stats']['mean']:.4f} ± {report['ocr_cer_stats']['std']:.4f}")
    print(f"Model CER:     {report['model_cer_stats']['mean']:.4f} ± {report['model_cer_stats']['std']:.4f}")
    print(f"CER Reduction: {report['relative_cer_reduction']*100:.2f}%\n")
    
    print("--- Word Error Rate (WER) ---")
    print(f"Original WER:  {report['ocr_wer_stats']['mean']:.4f} ± {report['ocr_wer_stats']['std']:.4f}")
    print(f"Model WER:     {report['model_wer_stats']['mean']:.4f} ± {report['model_wer_stats']['std']:.4f}")
    print(f"WER Reduction: {report['relative_wer_reduction']*100:.2f}%")
    print("="*50)


    os.makedirs("model_eval_logs", exist_ok=True)
    
    model_id = os.path.basename(os.path.normpath(args.model_path))
    dataset_id = os.path.splitext(os.path.basename(args.data_path))[0]
    meta_flag = "meta" if args.with_metadata else "nometa"
    
    filename = f"{model_id}_{dataset_id}_{meta_flag}_results.parquet"
    output_file = os.path.join("model_eval_logs", filename)
    
    df.to_parquet(output_file)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()