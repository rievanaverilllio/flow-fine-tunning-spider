# 🦙 Local LLaMA SQLCoder API (Windows)

Repositori ini membantu Anda:
- Fine-tuning LoRA untuk model LLaMA/SQLCoder di CPU (tanpa bitsandbytes)
- Merge hasil LoRA ke base model (HuggingFace format)
- Konversi ke GGUF untuk dipakai di llama.cpp
- Menjalankan REST API lokal dengan FastAPI untuk Text-to-SQL

----

## ✨ Fitur
- Jalankan model secara lokal, tanpa internet
- REST API via FastAPI, bisa diakses dari perangkat lain di jaringan yang sama
- Opsi expose publik (sementara) menggunakan LocalTunnel
- Parameter generasi dapat diatur (`max_tokens`, `temperature`, `stop`)

---

## � Struktur Utama
- `02_fine_tuning_with_checkpoint.py` — Fine-tuning LoRA (lanjut otomatis dari checkpoint terakhir)
- `03_merge_and_unload.py` — Merge LoRA ke base model (HF format)
- `llama.cpp/` — Alat konversi HF -> GGUF
- `model-gguf/merged-llama3-sql.Q8_0.gguf` — Contoh model hasil konversi/kuantisasi
- `05_llama_server.py` — REST API server (FastAPI + llama-cpp-python)
- `07_test_client.py` — Contoh client untuk memanggil API

---

## ✅ Persyaratan
- Windows + PowerShell
- Python 3.10 atau lebih baru
- (Untuk paket native) Visual Studio Build Tools disarankan jika pemasangan wheel gagal
- (Opsional) Node.js + npm jika ingin LocalTunnel

---

## 🔧 Setup Lingkungan
Jalankan di PowerShell pada folder proyek:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Catatan: `llama-cpp-python` membutuhkan ekstensi native. Jika gagal build, pasang Visual C++ Build Tools atau gunakan wheel yang cocok dengan versi Python Anda.

---

## 🏋️ Fine-tuning LoRA (opsional)
Skrip: `02_fine_tuning_with_checkpoint.py`

Konfigurasi yang perlu dicek di skrip:
- `MODEL_PATH` — path ke base model HF lokal
- Dataset Parquet: `dataset/train-00000-of-00001.parquet` dan `dataset/validation-00000-of-00001.parquet`

Jalankan:

```powershell
python .\02_fine_tuning_with_checkpoint.py
```

Output:
- Checkpoint tersimpan di `./hasil-finetune/checkpoint-*/`
- Adapter LoRA final tersimpan di `./llama-sql-adapter-final/`

Jika proses diulang, skrip otomatis melanjutkan dari checkpoint terbesar.

---

## 🔗 Merge LoRA ke Base Model (HF format)
Skrip: `03_merge_and_unload.py`

Pastikan variabel berikut benar:
- `base_model_path` — path base model HF
- `adapter_path` — `./llama-sql-adapter-final`
- `merged_path` — `./merged-llama3-sql`

Jalankan:

```powershell
python .\03_merge_and_unload.py
```

Hasil merge akan tersimpan sebagai model HF di folder `merged-llama3-sql/`.

---

## 🔁 Konversi HF -> GGUF (llama.cpp)
Gunakan script konversi di `llama.cpp/`. Lihat juga `04_convert_to_gguf.txt` untuk contoh lengkap.

Contoh (opsional, sesuaikan path model):

```powershell
python .\llama.cpp\convert_hf_to_gguf.py -i .\merged-llama3-sql -o .\model-gguf\merged-llama3-sql.gguf
```

Untuk kuantisasi (opsional), lihat dokumentasi llama.cpp atau gunakan alat kuantisasi bawaan untuk menghasilkan varian seperti `Q8_0`.

---

## 🚀 Jalankan REST API Server
Skrip: `05_llama_server.py`

Edit `model_path` agar menunjuk ke file GGUF Anda. Contoh di repo:

```python
model_path=r"D:\Vann\TA (SKRIPSI)\Project\Run-LLM\model-gguf\merged-llama3-sql.Q8_0.gguf"
```

Jalankan server:

```powershell
python .\05_llama_server.py
```

Server default di `http://0.0.0.0:8000`. Uji cepat dari mesin yang sama:

```powershell
Invoke-RestMethod -Method Post -ContentType 'application/json' -Uri http://127.0.0.1:8000/generate -Body (@{ prompt = 'contoh pertanyaan'; max_tokens = 128 } | ConvertTo-Json)
```

Untuk akses dari komputer lain satu jaringan, gunakan IP host (mis. `http://192.168.x.x:8000`) dan pastikan firewall mengizinkan port 8000.

---

## 🧪 Client Contoh
Skrip: `07_test_client.py`

Ubah `API_URL` jika memakai alamat berbeda (lokal atau LocalTunnel), lalu jalankan:

```powershell
python .\07_test_client.py
```

Contoh pemanggilan langsung via Python:

```python
import requests
url = "http://127.0.0.1:8000/generate"
data = {"prompt": "Tulis query untuk menghitung jumlah pegawai aktif", "max_tokens": 128}
print(requests.post(url, json=data, timeout=60).json())
```

---

## 🌐 Expose Publik (Opsional)

```powershell
npm install -g localtunnel
lt --port 8000 --subdomain llamaapi
```

Gunakan URL yang diberikan (mis. `https://llamaapi.loca.lt`) sebagai `API_URL` pada client.

---

## ❗ Troubleshooting
- Gagal install `llama-cpp-python`: pasang Visual C++ Build Tools atau gunakan wheel yang cocok.
- Server tidak start: cek `model_path` dan ketersediaan file `.gguf`.
- Respons lambat/timeout: kurangi `max_tokens`, set `temperature` rendah, atau pastikan spesifikasi mesin memadai.
- Import error saat fine-tuning: pastikan paket `transformers`, `peft`, `trl`, `datasets`, `accelerate` terpasang (lihat requirements).

---

## 📝 Lisensi
Hormat pada lisensi model dan repositori pihak ketiga (llama.cpp, Transformers). Gunakan sesuai ketentuan masing-masing.