from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import time

base_model_path = "D:\Vann\TA (SKRIPSI)\Project\Llama-3.2-3B-Instruct"
adapter_path = "./llama-sql-cot-adapter-cpu"
merged_path = "./merged-llama3-sql"

print("="*60)
print("🚀 MULAI PROSES MERGE MODEL LoRA KE BASE MODEL")
print("="*60)

# --- 1. Load Base Model & Tokenizer ---
print("\n📦 Memuat base model dan tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Tambahkan pad_token jika belum ada (agar sama dengan model hasil fine-tune)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

# Resize embeddings agar cocok
base_model.resize_token_embeddings(len(tokenizer))
print(f"✅ Tokenizer dan model siap. Vocab size: {len(tokenizer)}\n")

# --- 2. Load Adapter LoRA ---
print("🔧 Memuat adapter LoRA...")
for _ in tqdm(range(3), desc="Loading Adapter"):
    time.sleep(0.5)
model = PeftModel.from_pretrained(base_model, adapter_path)
print("✅ Adapter LoRA berhasil dimuat.\n")

# --- 3. Merge Adapter ke Base Model ---
print("🧠 Menggabungkan adapter ke base model...")
for _ in tqdm(range(5), desc="Merging Weights"):
    time.sleep(0.4)
merged_model = model.merge_and_unload()
print("✅ Merge selesai.\n")

# --- 4. Simpan Model Gabungan ---
print("💾 Menyimpan model hasil merge...")
for _ in tqdm(range(4), desc="Saving Model"):
    time.sleep(0.4)
merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print(f"✅ Model hasil merge berhasil disimpan di: {merged_path}\n")

print("="*60)
print("🎉 PROSES SELESAI — MODEL SUDAH DIGABUNGKAN DENGAN LoRA")
print("="*60)
