from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse
import os
import pandas as pd
import yaml
import torch

# Load BART config from YAML file
def load_config(file,model_type):
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    return config[model_type] # bart-base, bart-large,...


# Main function for fine-tuning BART
def main(args):
    # Load config
    config = load_config(args.config,args.model)

    # Select model
    model_name = f'facebook/{args.model}'
    output_dir = os.path.join('model', f'{args.output_name}')

    # Set up training data    
    train = Dataset.from_parquet(args.data)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess_function(examples):
        inputs = [str(doc) for doc in examples["ocr_text"]]
        targets = [str(doc) for doc in examples["ground_truth"]]
        
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
        labels = tokenizer(text_target=targets, max_length=1024, truncation=True, padding="max_length")
        
        masked_labels = []
        for label_array in labels["input_ids"]:
            masked_labels.append(
                [-100 if token == tokenizer.pad_token_id else token for token in label_array]
            )
            
        model_inputs["labels"] = masked_labels
        return model_inputs
    
    train = train.map(preprocess_function, batched=True,remove_columns=train.column_names)

    # Initialise BART
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    ## ========= I SPENT A WHOLE WEEK FOR THIS STUPID AH EXPLODING GRADIENT ERROR OF BART-LARGE REGARDING SOME FP16 FP32 PRECISION, FINALLY GEMINI GAVE ME A SOLUTION THAT TOUCHES THE MODEL AT THE BARE METAL LAYER. IF IT WORKS PLEASE DON'T TOUCH IT ==========
    # --- THE FP16 UNSCALE SILVER BULLET ---
    if args.model == "bart-large":
        # 1. Force the model weights natively to FP32 (in case the Hub loaded any in FP16)
        model = model.float()

        # 2. Inject the Backward Hook to intercept and cast gradients
        for param in model.parameters():
            if param.requires_grad:
                param.register_hook(lambda grad: grad.to(torch.float32) if grad is not None else None)
    # --------------------------------------
    
    # Fine-tune BART
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    config['learning_rate'] = float(config['learning_rate'])
    train_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        **config,
    )
    trainer = Seq2SeqTrainer(
        model,
        train_args,
        train_dataset=train,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    print("\n--- TRAINING CONFIGS ---")
    print("Model name:",model_name)
    print("Training data:",args.data)
    print("Output adapter will be saved to:",output_dir)
    print("\n--- TRAINING STARTED ---")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Fine tuned model saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning BART')
    parser.add_argument('--model', type=str, choices=['bart-base', 'bart-large', 'mbart-large-50'],
                        default='bart-base', help='Specify model: bart-base, bart-large')
    parser.add_argument('--output_name',type=str,default='bart-base-ocr',help='Output name of model adapter, always stored inside ./models')
    parser.add_argument('--config', type=str,default="finetune/config.yaml", help='Path to config')
    parser.add_argument('--data', type=str, default="data/hipe_aggregated_train.parquet", help='Path to training data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    main(args)
