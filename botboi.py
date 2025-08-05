from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
from ingest import nomnom
#import protobuf

#hf-auth-login

# Load the dataset 
nomnom()

dataset = load_dataset("text", data_files={"train": "training_data.txt"})

# Load the tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize Dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

tokenized_dataset = dataset["train"].map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfload16,
)

# Apply LoRA
lora_config = LoraConfig(
    r=16, # Low Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], #Mistral-specific modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./mixtral_lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=1, # lower duew to vram constraints
    gradient_accumulation_steps=8, # effective batch size = 8
    optim="adamw_torch",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=2e-5,
    fp16=True,
    max_grad_norm=0.3,
    warmup_ration=0.1,
)

# initialize and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
trainer.train()

# Save model
model.save_pretrained("./trained_mistral_lora_model")
tokenizer.save_pretrained("./trained_mistral_lora_model")


