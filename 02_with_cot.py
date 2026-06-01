import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ==============================
# 1. Path Model dan Dataset
# ==============================
MODEL_PATH = r"D:\Vann\TA (SKRIPSI)\Project\Llama-3.2-3B-Instruct"
TRAIN_FILE = "dataset/train-00000-of-00001.parquet"
VALID_FILE = "dataset/validation-00000-of-00001.parquet"
OUTPUT_DIR = "./hasil-finetune-cot"

# ==============================
# 2. Load Model & Tokenizer
# ==============================
print("Memuat model dan tokenizer (CPU)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,  # CPU only
    device_map="cpu",            # paksa CPU
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

tokenizer.padding_side = "right"

# ==============================
# 3. Konfigurasi LoRA (CPU-friendly)
# ==============================
print("Menerapkan LoRA ringan untuk CPU...")
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
print("Memuat dataset...")
dataset = load_dataset("parquet", data_files={'train': TRAIN_FILE, 'validation': VALID_FILE})

def preprocess_cot(sample):
    cot_prompt = f"""
### Pertanyaan:
{sample['question']}

### Penalaran:
1. Identifikasi tabel yang relevan.
2. Identifikasi kolom / entitas yang digunakan.
3. Tentukan operasi SQL yang sesuai (SELECT, JOIN, GROUP BY, AGGREGATE, dsb.)
4. Tulis SQL final berdasarkan langkah-langkah di atas.

### SQL:
{sample['query']}
"""
    return {"text": cot_prompt}

dataset = dataset.map(preprocess_cot, remove_columns=list(dataset['train'].features))
print("Contoh data:\n", dataset['train'][0]['text'])

# ==============================
# 5. Training Arguments CPU
# ==============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,    # CPU: batch 1
    gradient_accumulation_steps=4,    # akumulasi gradient agar efektif
    learning_rate=2e-5,
    logging_steps=50,
    save_total_limit=2,
    save_strategy="steps",
    save_steps=200,
    report_to="none",
    fp16=False,    # CPU tidak mendukung fp16
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
    max_seq_length=256,   # lebih pendek untuk CPU
    tokenizer=tokenizer,
    args=training_args,
)

# ==============================
# 7. Jalankan atau Lanjutkan Fine-Tuning
# ==============================
print("Mengecek keberadaan checkpoint...")

checkpoint_root = OUTPUT_DIR

# Cari folder yang namanya "checkpoint-xxx"
checkpoints = [
    d for d in os.listdir(checkpoint_root)
    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_root, d))
]

if checkpoints:
    # Urutkan berdasarkan angka checkpoint terbesar
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    last_checkpoint = os.path.join(checkpoint_root, checkpoints[-1])

    print(f"Melanjutkan fine-tuning dari checkpoint terakhir: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

else:
    print("Tidak ada checkpoint. Memulai fine-tuning baru...")
    trainer.train()


# ==============================
# 8. Simpan Adapter
# ==============================
adapter_dir = "./llama-sql-adapter-cpu"
trainer.model.save_pretrained(adapter_dir)
print(f"Fine-tuning selesai. Adapter disimpan di: {adapter_dir}")
