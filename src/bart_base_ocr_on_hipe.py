"""
Test BART on the HIPE OCRepair dataset with beam search and improved chunk stitching.
"""

import difflib
import argparse
import torch
from grab_samples import sample_jsonl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Colors and formatting
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"
YELLOW = "\033[93m"

def chunk_inference(text, model, tokenizer, chunk_size=256, overlap=32, device="cpu"):
    """
    Splits long text into chunks, processes them with Beam Search, 
    and stitches them back together.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    
    all_outputs = []
    
    # Process in chunks of tokens
    for i in range(0, len(input_ids), chunk_size - overlap):
        chunk = input_ids[i:i + chunk_size].unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Use Beam Search (standard for BART)
            output = model.generate(
                chunk, 
                max_new_tokens=chunk_size + 64,
                num_beams=5,
                early_stopping=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        all_outputs.append(decoded.strip())
        
    # Improved stitching: handle simple whitespace joining
    return " ".join(all_outputs)

def get_triple_snippets(ocr_text, gt_text, corrected_text, window=30, num_snippets_to_display=15):
    """
    Finds differences between GT and OCR, and shows how BART's corrected text compares.
    """
    # Normalize whitespace for better matching
    ocr_text = " ".join(ocr_text.split())
    gt_text = " ".join(gt_text.split())
    corrected_text = " ".join(corrected_text.split())

    matcher_ocr = difflib.SequenceMatcher(None, ocr_text, gt_text)
    matcher_corr = difflib.SequenceMatcher(None, corrected_text, gt_text)

    snippets = []

    for tag, i1, i2, j1, j2 in matcher_ocr.get_opcodes():
        if tag != 'equal':
            start_gt = max(0, j1 - window)
            end_gt = min(len(gt_text), j2 + window)

            pre_context = gt_text[start_gt:j1]
            post_context = gt_text[j2:end_gt]

            g_err = gt_text[j1:j2]
            o_err = ocr_text[i1:i2]

            c_start, c_end = 0, 0
            # Search for corresponding range in corrected_text
            for c_tag, ci1, ci2, cj1, cj2 in matcher_corr.get_opcodes():
                if cj1 <= j1 and cj2 >= j2:
                    c_start = ci1 + (j1 - cj1)
                    c_end = ci2 - (cj2 - j2)
                    break
                elif cj1 > j1:
                    c_start, c_end = ci1, ci1
                    break

            c_fix = corrected_text[c_start:c_end] if c_end > c_start else ""

            # Formatting snippets
            max_len = max(len(g_err), len(o_err), len(c_fix))
            g_padded = g_err.ljust(max_len)
            o_padded = o_err.ljust(max_len)
            c_padded = c_fix.ljust(max_len)

            gt_row  = f"...{pre_context}{GREEN}{BOLD}{g_padded}{RESET}{post_context}..."
            ocr_row = f"...{pre_context}{RED}{BOLD}{o_padded}{RESET}{post_context}..."
            cor_row = f"...{pre_context}{BLUE}{BOLD}{c_padded}{RESET}{post_context}..."

            snippets.append((tag.upper(), gt_row, ocr_row, cor_row))

            if len(snippets) >= num_snippets_to_display:
                break

    return snippets

def main():
    parser = argparse.ArgumentParser(description="Test BART on HIPE with Beam Search.")
    parser.add_argument("-i","--input_path",type=str,default="HIPE-OCRepair-2026-data/data/v0.9/icdar2017/en/hipe-ocrepair-bench_v0.9_icdar2017_v1.1_dev_en.jsonl",help="Path to the .jsonl data file")
    parser.add_argument("-n","--num_samples",type=int,default=1,help="Number of samples to grab")
    parser.add_argument("--seed",type=int,default=42,help="Seed for sampling")
    parser.add_argument("-w", "--window", type=int, default=25, help="Context characters")
    parser.add_argument("--chunk_size", type=int, default=256, help="BART chunk size")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model and tokenizer on {device}...")
    model = AutoModelForSeq2SeqLM.from_pretrained('pykale/bart-base-ocr').to(device)
    tokenizer = AutoTokenizer.from_pretrained('pykale/bart-base-ocr')

    try:
        samples = sample_jsonl(args.input_path, args.num_samples, args.seed)
        for i, item in enumerate(samples, 1):
            metadata = item.get("document_metadata", {})
            gt_transcript = item.get("ground_truth", {})['transcription_unit']
            ocr_transcript = item.get("ocr_hypothesis", {})['transcription_unit']

            print(f"Processing SAMPLE #{i} (Length: {len(ocr_transcript)} chars)...")
            corrected_transcript = chunk_inference(ocr_transcript, model, tokenizer, chunk_size=args.chunk_size, device=device)

            print(f"\n{YELLOW}{'='*85}{RESET}")
            print(f"{BOLD}SAMPLE #{i}{RESET} | ID: {metadata.get('document_id')} | Lang: {metadata.get('language')}")
            print(f"{YELLOW}{'='*85}{RESET}")

            snippets = get_triple_snippets(ocr_transcript, gt_transcript, corrected_transcript, args.window)

            if not snippets:
                print("No major OCR errors found in this sample.")
            else:
                for tag, gt_line, ocr_line, cor_line in snippets:
                    print(f"\n[{tag}]")
                    print(f"{BOLD}GT:   {RESET} {gt_line}")
                    print(f"{BOLD}OCR:  {RESET} {ocr_line}")
                    print(f"{BOLD}BART: {RESET} {cor_line}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
