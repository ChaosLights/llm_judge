import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "BAAI/JudgeLM-7B-v1.0"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

print("Model loaded!")