import json
import argparse
import random
import os

def sample_jsonl(input_path, n_samples, seed = 42):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} does not exist")
    
    with open(input_path,'r') as f:
        lines = f.readlines()
        
    sample_size = min(n_samples,len(lines))
    random.seed(seed)
    sampled_lines = random.sample(lines,sample_size)    
    
    return [json.loads(line) for line in sampled_lines]

def main():
    parser = argparse.ArgumentParser(description="Sample n entries from the provided jsonl file.")
    parser.add_argument("-i","--input_path",type=str,default="HIPE-OCRepair-2026-data/data/v0.9/dta19/de/hipe-ocrepair-bench_v0.9_dta19-l0_v0.1_dev_de.jsonl",help="Path to the .jsonl data file")
    parser.add_argument("-n","--num_samples",type=int,default=5,help="Number of samples to grab (default: 5)")
    parser.add_argument("--seed",type=int,default=42,help="it doesnt matter, hear me out, 42 is the only seed we need")
    
    args = parser.parse_args()
    
    try:
        samples = sample_jsonl(args.input_path,args.num_samples,args.seed)
        
        for i,item in enumerate(samples,1):
            metadata = item.get("document_metadata", {})
            gt = item.get("ground_truth", {})
            
            ocr = item.get("ocr_hypothesis", {})
            cer_hypo = ocr["quality_report"]["cer"]
            wer_hypo = ocr["quality_report"]["wer"]
            
            print(f"Sample #{i} | ID: {metadata.get('document_id')}")
            print(f"Language: {metadata.get('language')} | Date: {metadata.get('date')}")
            print(f"GT:  {gt.get('transcription_unit')[:100]}...")
            print(f"OCR: {ocr.get('transcription_unit')[:100]}...")
            print(f"Metrics: CER = {cer_hypo:.4f}, WER = {wer_hypo}")
            print("-" * 50)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()