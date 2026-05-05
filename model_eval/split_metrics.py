import pandas as pd
import os
import glob
from metrics import get_comparative_report

def analyze_aggregated_results(file_path):
    print(f"\n" + "="*60)
    print(f"ANALYZING: {os.path.basename(file_path)}")
    print("="*60)
    
    df = pd.read_parquet(file_path)
    
    # 1. Evaluate the 5 Datasets
    print("\n--- RESULTS BY DATASET ---")
    if 'dataset' in df.columns:
        for dataset_name in df['dataset'].dropna().unique():
            subset = df[df['dataset'] == dataset_name]
            
            # Skip if subset is empty
            if len(subset) == 0: continue
                
            report = get_comparative_report(
                subset['ground_truth'].tolist(),
                subset['ocr_text'].tolist(),
                subset['model_output'].tolist()
            )
            cer_red = report['relative_cer_reduction'] * 100
            wer_red = report['relative_wer_reduction'] * 100
            print(f"{dataset_name:<20} | CER Red: {cer_red:>7.2f}% | WER Red: {wer_red:>7.2f}% | (Samples: {len(subset)})")
    else:
        print("No 'dataset' column found in this file.")

    # 2. Evaluate the 3 Languages
    print("\n--- RESULTS BY LANGUAGE ---")
    if 'language' in df.columns:
        for lang in df['language'].dropna().unique():
            subset = df[df['language'] == lang]
            
            if len(subset) == 0: continue
            
            report = get_comparative_report(
                subset['ground_truth'].tolist(),
                subset['ocr_text'].tolist(),
                subset['model_output'].tolist()
            )
            cer_red = report['relative_cer_reduction'] * 100
            wer_red = report['relative_wer_reduction'] * 100
            print(f"{lang:<20} | CER Red: {cer_red:>7.2f}% | WER Red: {wer_red:>7.2f}% | (Samples: {len(subset)})")
    else:
        print("No 'language' column found in this file.")

def main():
    # Find all the aggregated test results in your logs folder
    log_dir = "model_eval_logs"
    
    # This grabs all files that were evaluated on the hipe_aggregated test set
    search_pattern = os.path.join(log_dir, "*hipe_aggregated*results.parquet")
    aggregated_files = glob.glob(search_pattern)
    
    if not aggregated_files:
        print(f"No aggregated result files found in {log_dir}/")
        return

    print(f"Found {len(aggregated_files)} aggregated model evaluations. Splitting metrics...")
    
    for file_path in aggregated_files:
        analyze_aggregated_results(file_path)

if __name__ == "__main__":
    main()