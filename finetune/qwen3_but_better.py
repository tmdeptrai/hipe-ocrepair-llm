import os
import json
import yaml
import argparse
import pandas as pd
import torch
from datasets import Dataset
from pydantic import BaseModel, Field
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ==============================================================================
# Schema Definition
# ==============================================================================
class OCRCorrection(BaseModel):
    corrected_text: str = Field(description="The cleaned and corrected version of the historical OCR text.")
    has_modifications: bool = Field(description="True if the text was changed, False if no errors were found.")

# ==============================================================================
# Helper Functions
# ==============================================================================
def load_config(file: str, model_type: str) -> dict:
    with open(file, 'r') as f:
        return yaml.safe_load(f)[model_type]

def create_formatting_func(tokenizer, with_metadata=False):
    schema_str = json.dumps(OCRCorrection.model_json_schema(), indent=2)
    
    def formatting_func(example):
        ocr_text = example['ocr_text']
        ground_truth = example['ground_truth']
        was_changed = ocr_text.strip() != ground_truth.strip()
        
        target_obj = OCRCorrection(
            corrected_text=ground_truth,
            has_modifications=was_changed
        )
        target_json = target_obj.model_dump_json()

        if with_metadata:
            dataset = example.get('dataset', 'Unknown')
            lang = example.get('language', 'Unknown')
            year = example.get('year', None)
            
            year_str = str(int(year)) if pd.notnull(year) else 'Unknown'

            instruction = f"""You are an expert OCR post-correction system for historical newspapers in many languages.

Your task is to correct the OCR hypothesis while preserving the original historical wording, spelling, punctuation style, paragraph flow, and meaning as much as possible.

You are given the document metadata and the OCR hypothesis.

You must output only the corrected transcription unit formatted as JSON.

Important principles:
- Correct OCR errors caused by character confusion, broken words, incorrect punctuation, spacing errors, and misrecognized letters.
- Preserve historical spelling and period-specific language.
- Do not modernize vocabulary.
- Do not rewrite sentences for fluency.
- Do not paraphrase.
- Do not add information that is not supported by the OCR text.
- Do not remove content unless it is clearly an OCR artefact.
- Keep the text as one continuous transcription unit unless the input clearly contains paragraph breaks.
- Preserve names, initials, titles, and abbreviations carefully.
- Preserve capitalization when historically plausible.
- Be especially careful with 19th-century punctuation, long sentences, hyphenated compounds, and editorial address forms such as "Mr. Editor" or "Sir".

Document context:
- Dataset: {dataset}
- Language: {lang}
- Year: {year_str}

Expected correction behavior for this document:
The OCR hypothesis comes from a historical newspaper passage in {lang}. It contains many typical OCR errors:
- Misread letters: "di position" → "disposition", "phrenotogized" → "phrenologized"
- Confused punctuation: "R '" → "R.", "prove '" → "prove ?"
- Misread words: "nliat docs" → "what does", "site" → "she", "tire" → "the"
- Broken or merged words: "thesameroutineof" → "the same routine of"
- Wrong substitutions caused by old print: "lie" may actually be "be", "tlie" may be "the", "tlmt" may be "that"
- Hyphenation and line-break artefacts: repair only when clearly caused by OCR
- Historical typography artefacts: normalize obvious OCR artefacts but preserve meaningful punctuation and hyphenated forms

Use sentence-level structure as a soft guide only. If the OCR sentence offsets are wrong or split the text badly, prioritize the natural corrected text.

Correction strategy:
1. Read the full OCR hypothesis before correcting.
2. Infer intended words from local context and historical newspaper style.
3. Correct high-confidence OCR errors.
4. Leave uncertain historical or rare words unchanged unless the OCR form is clearly impossible.
5. Preserve the author's rhetorical style, including irony, long sentences, and formal address.
6. Do not hallucinate missing passages.

Return ONLY a JSON object matching this schema:
{schema_str}"""

        else:
            # Base instruction for runs without metadata enabled
            instruction = f"""You are an expert OCR post-correction system.
Correct the errors in the following historical OCR text following strict historical preservation guidelines.
Return ONLY a JSON object matching this schema:
{schema_str}"""

        # Map to the ChatML format Qwen expects
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"OCR hypothesis:\n{ocr_text}"},
            {"role": "assistant", "content": target_json}
        ]
        
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    
    return formatting_func

# ==============================================================================
# Main Training Logic
# ==============================================================================
def main(args):
    config = load_config(args.config, args.model)
    model_name = f'Qwen/{args.model}'
    output_dir = os.path.join('model', f'{args.output_name}-{"meta" if args.with_metadata else "base"}')

    # Data loading
    train_df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data)
    train_dataset = Dataset.from_pandas(train_df)

    # 4-bit Quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Optimized LoRA parameters
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,          
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules="all-linear",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        use_cache=False,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = 'right'

    formatting_func = create_formatting_func(tokenizer, with_metadata=args.with_metadata)

    if 'learning_rate' in config:
        config['learning_rate'] = float(config['learning_rate'])

    # SFT Trainer Configuration
    train_args = SFTConfig(
        output_dir=output_dir,
        max_length=2048,
        dataset_kwargs={
            "add_special_tokens": False,  
            "append_concat_token": False,
        },
        **config,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    
    print("\n====== Training Configuration ======")
    print(f"Model: {model_name}")
    print(f"Data: {args.data} ({len(train_dataset)} samples)")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n====== Training Started ======")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel and tokenizer successfully saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3-8B')
    parser.add_argument('--config', type=str, default='finetune/config.yaml')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output_name', type=str, default='qwen3-8B-ocr-improv1')
    parser.add_argument('--with_metadata', action='store_true')
    
    main(parser.parse_args())