import torch
import pandas as pd
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from metrics import get_comparative_report

def batch_inference(texts: list, model, tokenizer, device, batch_size=16):
    """
    Performs inference in batches for massive speedups on GPU.
    Relies on the fact that the dataset is already chunked below 1024 tokens.
    """
    all_outputs = []
    
    # Process the dataset in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Corrections"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize with truncation and padding
        inputs = tokenizer(
            batch_texts, 
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
        all_outputs.extend([text.strip() for text in decoded])
        
    return all_outputs

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned BART on HIPE chunks.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned BART model")
    parser.add_argument("--data_path", type=str, default="data/hipe_aggregated_test.parquet", help="Path to test parquet")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for quick testing")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(args.device)
    
    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    
    if args.limit:
        df = df.head(args.limit)
    
    print(f"Running batch inference on {len(df)} samples...")
    # Run the entire OCR text column through the batched inference
    corrected_texts = batch_inference(
        df['ocr_text'].tolist(), 
        model, 
        tokenizer, 
        args.device, 
        batch_size=args.batch_size
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
    
    print(f"Original CER:  {report['ocr_stats']['mean']:.4f} ± {report['ocr_stats']['std']:.4f}")
    print(f"Model CER:     {report['model_stats']['mean']:.4f} ± {report['model_stats']['std']:.4f}")
    print(f"CER Reduction: {report['relative_cer_reduction']*100:.2f}%")
    print(f"Cohen's d:     {report['cohens_d']:.4f}")

    os.makedirs("model_eval_logs", exist_ok=True)
    output_file = os.path.join("model_eval_logs", f"{str(args.model_path).replace('/', '_')}_eval_results.parquet")
    df.to_parquet(output_file)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()