import os
import random
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch

MODEL_NAME = "gpt2"
DATA_PATH = "dnd_treasures.txt"
OUTPUT_DIR = "./dnd_treasure_gpt2"
SPLIT_RATIO = 0.95
EPOCHS = 3
BATCH_SIZE = 2

# 1. Load and split data
with open(DATA_PATH, encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
random.shuffle(lines)
split_idx = int(len(lines) * SPLIT_RATIO)
train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

train_file = "train.txt"
val_file = "val.txt"
with open(train_file, "w", encoding="utf-8") as f:
    f.write("\n".join(train_lines))
with open(val_file, "w", encoding="utf-8") as f:
    f.write("\n".join(val_lines))

# 2. Load tokenizer and model
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Set pad_token to eos_token for GPT-2
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# 3. Prepare datasets
block_size = 128

def load_text_dataset(file_path, tokenizer, block_size=128):
    return Dataset.from_dict({"text": [line for line in open(file_path, encoding="utf-8")]})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=block_size)

train_dataset = load_text_dataset(train_file, tokenizer)
val_dataset = load_text_dataset(val_file, tokenizer)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="epoch",
    logging_steps=100,
    prediction_loss_only=False,
    report_to=[],
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 6. Train
train_result = trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 7. Save training metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state() 