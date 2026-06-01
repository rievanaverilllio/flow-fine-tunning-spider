import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ==============================
# 1. Model and Dataset Paths
# ==============================
MODEL_PATH = r"D:\Vann\TA (SKRIPSI)\Project\Llama-3.2-3B-Instruct"
TRAIN_FILE = "dataset/train-00000-of-00001.parquet"
VALID_FILE = "dataset/validation-00000-of-00001.parquet"
OUTPUT_DIR = "./finetune-output"

# ==============================
# 2. Load Model & Tokenizer
# ==============================
print("Loading model and tokenizer (CPU)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,  # CPU only
    device_map="cpu",            # force CPU
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

tokenizer.padding_side = "right"

# ==============================
# 3. LoRA configuration (CPU-friendly)
# ==============================
print("Applying lightweight LoRA for CPU...")
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==============================
# 4. Load & Preprocess Dataset
# ==============================
print("Loading dataset...")
dataset = load_dataset("parquet", data_files={'train': TRAIN_FILE, 'validation': VALID_FILE})

def preprocess(sample):
    prompt = f"""
### Question:
{sample['question']}

### SQL:
{sample['query']}
"""
    return {"text": prompt}

dataset = dataset.map(preprocess, remove_columns=list(dataset['train'].features))
print("Sample data:\n", dataset['train'][0]['text'])

# ==============================
# 5. Training Arguments (CPU)
# ==============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,    # CPU: batch size 1
    gradient_accumulation_steps=4,    # gradient accumulation for efficiency
    learning_rate=2e-5,
    logging_steps=50,
    save_total_limit=2,
    save_strategy="steps",
    save_steps=100,
    report_to="none",
    fp16=False,    # CPU does not support fp16
    bf16=False,
    load_best_model_at_end=False,
)

# ==============================
# 6. SFT Trainer
# ==============================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=256,   # shorter sequence length for CPU
    tokenizer=tokenizer,
    args=training_args,
)

# ==============================
# 7. Run or Resume Fine-Tuning
# ==============================
print("Checking for existing checkpoints...")

checkpoint_root = OUTPUT_DIR

# Search for folders named "checkpoint-xxx"
checkpoints = [
    d for d in os.listdir(checkpoint_root)
    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_root, d))
]

if checkpoints:
    # Urutkan berdasarkan angka checkpoint terbesar
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    last_checkpoint = os.path.join(checkpoint_root, checkpoints[-1])

    print(f"Resuming fine-tuning from last checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

else:
    print("No checkpoint found. Starting new fine-tuning...")
    trainer.train()


# ==============================
# 8. Save Adapter
# ==============================
adapter_dir = "./llama-sql-adapter-cpu"
trainer.model.save_pretrained(adapter_dir)
print(f"Fine-tuning complete. Adapter saved to: {adapter_dir}")
