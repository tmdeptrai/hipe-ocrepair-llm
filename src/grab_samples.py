"""
Grab random samples from an input jsonl data file, display metadata, CER and WER metrics
show differences between GT text, OCR-ed text
"""

import json
import argparse
import random
import os
import difflib

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
BOLD = "\033[1m"
YELLOW = "\033[93m"

def sample_jsonl(input_path, n_samples, seed = 42):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} does not exist")
    
    with open(input_path,'r') as f:
        lines = f.readlines()
        
    sample_size = min(n_samples,len(lines))
    random.seed(seed)
    sampled_lines = random.sample(lines,sample_size)    
    
    return [json.loads(line) for line in sampled_lines]

def get_aligned_snippets(ocr_text, gt_text, window=30, num_snippets_to_display=5):
    """
    Finds differences between GT and OCR
    window: the text surrounding it, default = 30
    num_snippets_displayed: avoid displaying too many differences, -1 = display all
    """
    
    matcher = difflib.SequenceMatcher(None, ocr_text, gt_text)
    snippets = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            # surrounding window of text
            start_ocr = max(0, i1 - window)
            end_ocr = min(len(ocr_text), i2 + window)
            
            # extract fragments
            pre_context = ocr_text[start_ocr:i1].replace('\n', ' ')
            post_context = ocr_text[i2:end_ocr].replace('\n', ' ')
            
            o_err = ocr_text[i1:i2].replace('\n', ' ⏎ ')
            g_fix = gt_text[j1:j2].replace('\n', ' ⏎ ')

            # align the error and the fix
            max_len = max(len(o_err), len(g_fix))
            o_padded = o_err.ljust(max_len)
            g_padded = g_fix.ljust(max_len)

            # construct the two rows
            gt_row = f"...{pre_context}{GREEN}{BOLD}{g_padded}{RESET}{post_context}..."
            ocr_row = f"...{pre_context}{RED}{BOLD}{o_padded}{RESET}{post_context}..."
            
            snippets.append((tag.upper(), gt_row, ocr_row))
    
    num_snippets_remaining = len(snippets) - num_snippets_to_display
    snippets = snippets[:num_snippets_to_display] if num_snippets_to_display != -1 else snippets
    
    return snippets,num_snippets_remaining


def main():
    parser = argparse.ArgumentParser(description="Sample n entries from the provided jsonl file.")
    parser.add_argument("-i","--input_path",type=str,default="HIPE-OCRepair-2026-data/data/v0.9/icdar2017/en/hipe-ocrepair-bench_v0.9_icdar2017_v1.1_dev_en.jsonl",help="Path to the .jsonl data file")
    parser.add_argument("-n","--num_samples",type=int,default=5,help="Number of samples to grab (default: 5)")
    parser.add_argument("--seed",type=int,default=42,help="it doesnt matter, hear me out, 42 is the only seed we need")
    parser.add_argument("-w", "--window", type=int, default=25, help="Context characters")

    args = parser.parse_args()
    
    try:
        samples = sample_jsonl(args.input_path,args.num_samples,args.seed)
        
        for i,item in enumerate(samples,1):
            metadata = item.get("document_metadata", {})
            gt = item.get("ground_truth", {})
            
            ocr = item.get("ocr_hypothesis", {})
            
            print(f"\n{YELLOW}{'='*85}{RESET}")
            print(f"{BOLD}SAMPLE #{i}{RESET} | ID: {metadata.get('document_id')} | Lang: {metadata.get('language')}")
            print(f"CER: {ocr['quality_report']['cer']:.4f} | WER: {ocr['quality_report']['wer']:.4f}")
            print(f"{YELLOW}{'='*85}{RESET}")

            snippets,remaining = get_aligned_snippets(ocr['transcription_unit'], gt['transcription_unit'], args.window)
            
            if not snippets:
                print("No errors found.")
            else:
                for tag, gt_line, ocr_line in snippets:
                    print(f"\n[{tag}]")
                    print(f"{BOLD}GT: {RESET} {gt_line}")
                    print(f"{BOLD}OCR:{RESET} {ocr_line}")

            print(f"\n(and {remaining} more...)")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()