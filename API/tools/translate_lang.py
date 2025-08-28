from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Check device availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to device
model = model.to(device)

def main():
    inputs ="en: We're on a jorney to advance and demoratize artificial intelligece through open source and open science."


    outputs = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to(device), max_length=512)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for res in result:
        print(res)

if __name__ == "__main__":
    main()

