"""
Model card: https://huggingface.co/pykale/bart-base-ocr
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained('pykale/bart-base-ocr')
tokenizer = AutoTokenizer.from_pretrained('pykale/bart-base-ocr')

ocr = "The defendant wits'fined �5 and costs. Dreadſul weather on tho coast."
print(f"Original: {ocr}")

inputs = tokenizer(ocr, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=1024)

print(f"Corrected: {tokenizer.batch_decode(outputs,skip_special_tokens=True)[0]}")
