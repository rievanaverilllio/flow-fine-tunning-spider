import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ==============================
# 1. Path Model dan Dataset
# ==============================
MODEL_PATH = r"D:\Vann\TA (SKRIPSI)\Project\Llama-3.2-3B-Instruct"
TRAIN_FILE = "dataset/train-00000-of-00001.parquet"
VALID_FILE = "dataset/validation-00000-of-00001.parquet"
OUTPUT_DIR = "./hasil-finetune-cot"
SAVE_STEPS = 100

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
### Question:
{sample['question']}

### Reasoning:
1. Identify the relevant tables.
2. Identify the columns / entities used.
3. Determine the appropriate SQL operations (SELECT, JOIN, GROUP BY, AGGREGATE, etc.)
4. Write the final SQL based on the steps above.

### SQL:
{sample['query']}
"""
    return {"text": cot_prompt}

dataset = dataset.map(preprocess_cot, remove_columns=list(dataset['train'].features))
print("Contoh data:\n", dataset['train'][0]['text'])


def load_loss_history(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file_handle:
            try:
                history = json.load(file_handle)
                return history if isinstance(history, list) else []
            except json.JSONDecodeError:
                return []
    return []


def save_loss_history(file_path, history):
    with open(file_path, "w", encoding="utf-8") as file_handle:
        json.dump(history, file_handle, indent=2)


class LossHistoryCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.train_loss_file = os.path.join(output_dir, "train_loss.json")
        self.validation_loss_file = os.path.join(output_dir, "validation_loss.json")
        self.train_loss_history = load_loss_history(self.train_loss_file)
        self.validation_loss_history = load_loss_history(self.validation_loss_file)
        self.pending_logs = {}

    def _flush_if_ready(self, step):
        pending = self.pending_logs.get(step, {})
        train_loss = pending.get("train_loss")
        validation_loss = pending.get("validation_loss")

        if train_loss is not None and validation_loss is not None:
            print(f"[step {step}] train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}")
            self.train_loss_history.append({"step": step, "loss": train_loss})
            self.validation_loss_history.append({"step": step, "loss": validation_loss})
            save_loss_history(self.train_loss_file, self.train_loss_history)
            save_loss_history(self.validation_loss_file, self.validation_loss_history)
            del self.pending_logs[step]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return control

        step = state.global_step
        self.pending_logs.setdefault(step, {})["train_loss"] = float(logs["loss"])
        self._flush_if_ready(step)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return control

        step = state.global_step
        self.pending_logs.setdefault(step, {})["validation_loss"] = float(metrics["eval_loss"])
        self._flush_if_ready(step)
        return control

# ==============================
# 5. Training Arguments CPU
# ==============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,    # CPU: batch 1
    gradient_accumulation_steps=4,    # akumulasi gradient agar efektif
    learning_rate=2e-5,
    logging_steps=SAVE_STEPS,
    save_total_limit=2,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
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
    callbacks=[LossHistoryCallback(OUTPUT_DIR)],
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
