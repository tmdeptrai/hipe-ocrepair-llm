from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from pydantic import BaseModel, Field
import argparse
import os
import pandas as pd
import torch
import yaml
import json


# Define the output schema using Pydantic
class OCRCorrection(BaseModel):
    corrected_text: str = Field(description="The cleaned and corrected version of the OCR text.")
    has_modifications: bool = Field(description="True if the text was changed, False if no errors were found.")


# Load Qwen3 config from YAML file
def load_config(file, model_type):
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    return config[model_type]


def create_formatting_func(tokenizer, with_metadata=False):
    """
    Creates a formatting function for the SFTTrainer that enforces JSON output
    matching the Pydantic OCRCorrection schema.
    """
    # Generate the JSON schema string once
    schema_str = json.dumps(OCRCorrection.model_json_schema(), indent=2)
    
    # Notice we changed 'examples' to 'example' to reflect the unbatched input
    def formatting_func(example):
        # No more loop needed! Extract strings directly.
        ocr_text = example['ocr_text']
        ground_truth = example['ground_truth']
        
        # Determine if modifications were made
        was_changed = ocr_text.strip() != ground_truth.strip()
        
        # Create the target JSON string
        target_obj = OCRCorrection(
            corrected_text=ground_truth,
            has_modifications=was_changed
        )
        target_json = target_obj.model_dump_json()

        if with_metadata:
            # Safely get metadata for the single row
            year = example.get('year', None)
            dataset = example.get('dataset', 'unknown')
            lang = example.get('language', 'unknown')
            
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

        messages = [
            {"role": "user", "content": f"{instruction}\n\nOCR Text:\n{ocr_text}"},
            {"role": "assistant", "content": target_json}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # SFTTrainer expects a string (or list of strings) to be returned
        return text
    
    return formatting_func


def main(args):
    config = load_config(args.config, args.model)
    model_name = f'Qwen/{args.model}'
    output_dir = os.path.join('model', f'{args.output_name}-{"meta" if args.with_metadata else "base"}')

    if args.data.endswith('.parquet'):
        train_df = pd.read_parquet(args.data)
    else:
        train_df = pd.read_csv(args.data)
    
    train_dataset = Dataset.from_pandas(train_df)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
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

    if 'learning_rate' in config:
        config['learning_rate'] = float(config['learning_rate'])
    
    formatting_func = create_formatting_func(tokenizer, with_metadata=args.with_metadata)

    train_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
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
    
    print("\n======Training configs========")
    print(f"Model name: {model_name}")
    print(f"Training data: {args.data}")
    print("Hyperparameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\n=======Training started=========")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"\nModel has been saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Instruction-tuning Qwen 3 with Pydantic JSON enforcement')
    parser.add_argument('--model', type=str, choices=['qwen3-4B', 'qwen3-8B'],
                        default='qwen3-4B', help='Specify model: qwen3-4B, qwen3-8B')
    parser.add_argument('--config', type=str, default='finetune/config.yaml', help='Path to config')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output_name', type=str, default='qwen3-4B-ocr', help='Output Name of the adapter, e.g: qwen3-4B-ocr')
    parser.add_argument('--with_metadata', action='store_true', help='Include metadata in prompts')
    
    args = parser.parse_args()
    main(args)
