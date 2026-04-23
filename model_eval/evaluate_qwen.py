import torch
import pandas as pd
import argparse
import os
import json
import re
from tqdm import tqdm
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from metrics import get_comparative_report

class OCRCorrection(BaseModel):
    corrected_text: str = Field(description="The cleaned and corrected version of the OCR text.")
    has_modifications: bool = Field(description="True if the text was changed, False if no errors were found.")

def extract_corrected_text(json_str: str, fallback_text: str) -> str:
    """Safely attempts to parse the JSON and extract the corrected text."""
    # Strip markdown blocks if the model wrapped it in code ticks
    cleaned_str = json_str.replace("```json", "").replace("```", "").strip()
    
    # Remove the <think> tags and everything inside them
    # flags=re.DOTALL is crucial so it catches newlines (\n) inside the think block
    cleaned_str = re.sub(r'<think>.*?</think>', '', cleaned_str, flags=re.DOTALL).strip()
    
    # Isolate the JSON object
    start_idx = cleaned_str.find('{')
    end_idx = cleaned_str.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        cleaned_str = cleaned_str[start_idx:end_idx+1]
    
    try:
        data = json.loads(cleaned_str,strict=False)
        return data.get("corrected_text", fallback_text)
    except json.JSONDecodeError as e:
        print("\n" + "!"*40)
        print(f"JSON PARSE ERROR: {e}")
        print("--- RAW MODEL OUTPUT START ---")
        print(repr(cleaned_str)) 
        print("--- RAW MODEL OUTPUT END ---")
        print("!"*40 + "\n")
        return fallback_text

def batch_inference(df: pd.DataFrame, model, tokenizer, device, batch_size=4, with_metadata=False):
    all_outputs = []
    schema_str = json.dumps(OCRCorrection.model_json_schema(), indent=2)
    
    for i in tqdm(range(0, len(df), batch_size), desc="Generating Corrections"):
        batch_df = df.iloc[i:i + batch_size]
        batch_texts = batch_df['ocr_text'].tolist()
        
        prompts = []
        for _, row in batch_df.iterrows():
            ocr_text = row['ocr_text']
            
            if with_metadata:
                year = row.get('year', None)
                dataset = row.get('dataset', 'unknown')
                lang = row.get('language', 'unknown')
                
                context = f"Dataset: {dataset}, Language: {lang}"
                if year and pd.notnull(year):
                    context += f", Year: {int(year)}"
                
                instruction = (
                    f"Perform OCR post-correction given this context: {context}.\n"
                    f"Return only a JSON object matching this schema:\n{schema_str}"
                )
            else:
                instruction = (
                    "Correct the errors in the following OCR text.\n"
                    f"Return only a JSON object matching this schema:\n{schema_str}"
                )

            messages = [{"role": "user", "content": f"{instruction}\n\nOCR Text:\n{ocr_text}"}]
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts.append(prompt)
            
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1024,
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id
            )
            
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_length:]
        
        decoded_jsons = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for orig_text, json_output in zip(batch_texts, decoded_jsons):
            final_text = extract_corrected_text(json_output, fallback_text=orig_text)
            all_outputs.append(final_text.strip())
            
    return all_outputs

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned Qwen3 on HIPE chunks.")
    parser.add_argument("--base_model", type=str, default="Qwen/qwen3-4B", help="HuggingFace base model ID")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to your saved LoRA adapter")
    parser.add_argument("--data_path", type=str, default="data/hipe_aggregated_test.parquet", help="Path to test parquet")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--with_metadata", action="store_true", help="Include metadata in evaluation prompts")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for quick testing")
    
    args = parser.parse_args()
    
    print(f"Loading Base Model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' 
        
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    
    if args.adapter_path:
        print(f"Loading LoRA Adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        print("No adapter path provided. Evaluating the BASE MODEL...")
        model = base_model
        
    model.eval()
    
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
    
    df['model_output'] = corrected_texts
    
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
    if args.adapter_path:
        model_id = os.path.basename(os.path.normpath(args.adapter_path))
    else:
        model_id = f"zeroshot_{os.path.basename(os.path.normpath(args.base_model))}"
        
    dataset_id = os.path.splitext(os.path.basename(args.data_path))[0]
    meta_flag = "meta" if args.with_metadata else "nometa"
    filename = f"{model_id}_{dataset_id}_{meta_flag}_results.parquet"
    output_file = os.path.join("model_eval_logs", filename)
    
    df.to_parquet(output_file)
    print(f"\nDetailed results saved to {output_file}")
if __name__ == "__main__":
    main()