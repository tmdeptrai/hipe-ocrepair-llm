from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import argparse
import os
import pandas as pd
import torch
import yaml


# Load Qwen3 config from YAML file
def load_config(file,model_type):
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
    return config[model_type] # qwen3-4B, qwen3-8B,...


# Main function for instruction-tuning Qwen3
def main(args):
    # Load config
    config = load_config(args.config)

    # Select model
    model_name = f'qwen/{args.model}'
    output_dir = os.path.join('model', f'{args.model}-ocr')

    # Set up training data
    train = pd.read_csv(args.data)
    train = Dataset.from_pandas(train)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # LoRA config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias='none',
        task_type='CAUSAL_LM',
        # MAYBE ADD: target_modules="all-linear"
    )

    # Initialise Qwen3
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        use_cache=False,
        device_map='auto',
    )
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    config['learning_rate'] = float(config['learning_rate'])
    train_args = SFTConfig(
        output_dir=output_dir,
        **config,
    )
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train,
        peft_config=peft_config,
        max_seq_length=1024,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=prompt_template,
    )
    trainer.train()
    trainer.save_model(output_dir)


if __name__ == '__main__':
    # Parse arguments for model/config/data
    parser = argparse.ArgumentParser(description='Instruction-tuning Qwen 3')
    parser.add_argument('--model', type=str, choices=['qwen3-4B', 'qwen3-8B'],
                        default='qwen3-4B', help='Specify model: qwen3-4B, qwen3-8B')
    parser.add_argument('--config', type=str, help='Path to config')
    parser.add_argument('--data', type=str, help='Path to training data')
    args = parser.parse_args()

    main(args)


""" OFFICAL CODE SNIPET OF QWEN/QWEN3 TO RUN AN EXAMPLE
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)


"""