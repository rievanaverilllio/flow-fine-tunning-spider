from llama_cpp import Llama
from fastapi import FastAPI, Request
import uvicorn

# =======================
# 1. LOAD MODEL LOKAL
# =======================
llm = Llama(
    model_path=r"D:\Vann\TA (SKRIPSI)\Project\Run-LLM\model-gguf\merged-llama3-sql.Q8_0.gguf",  # Gunakan raw string (r"...") agar backslash tidak error
    n_ctx=4096,       # Panjang konteks (naikkan biar prompt panjang tidak terpotong)
    n_threads=8,      # Jumlah thread CPU
    n_gpu_layers=0,   # 0 = CPU-only
    verbose=False     # Supaya log tidak spam di console
)

# =======================
# 2. BUAT API SERVER
# =======================
app = FastAPI(title="Local LLaMA API")

@app.get("/")
def home():
    return {"message": "✅ LLaMA local API aktif 🚀"}

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    user_prompt = data.get("prompt", "").strip()
    max_tokens = int(data.get("max_tokens", 200))

    # Jika prompt kosong
    if not user_prompt:
        return {"error": "Prompt tidak boleh kosong."}

    # Template prompt baru dengan konteks tabel sebenarnya
    formatted_prompt = f"""
You are an expert AI assistant specialized in generating SQL queries for the table below.
Use the correct column names and conditions, and do not repeat filters unnecessarily.

### Table: employee_data

Columns description:
- emp_id: Unique numeric employee identifier (e.g., 1001)
- first_name: Employee's first name (e.g., Susan)
- last_name: Employee's last name (e.g., Exantus)
- start_date: The date when the employee started working (e.g., 2019-08-29)
- exit_date: The date when the employee left the company (NULL if still active)
- title: Job title (e.g., Software Engineer)
- supervisor: Name of the employee's supervisor (e.g., Angela Carlson)
- ad_email: Employee's company email (e.g., susan.exantus@bilearner.com)
- business_unit: Business process center or division code (e.g., BPC)
- employee_status: Employment status (e.g., Active, Terminated)
- employee_type: Type of employment (e.g., Part-Time, Full-Time)
- pay_zone: Pay scale zone (e.g., Zone A, Zone B)
- employee_classification_type: Classification type (e.g., Part-Time, Full-Time)
- termination_type: Termination type (e.g., Unk, Voluntary, Involuntary)
- termination_description: Description of termination reason (nullable)
- department_type: Department category (e.g., Software Engineering)
- division: Division name (e.g., Engineers)
- dob: Date of birth (e.g., 1957-09-21)
- state: Work state abbreviation (e.g., MA)
- job_function_description: Job role description (e.g., Engineer)
- gender_code: Gender (e.g., Female, Male)
- location_code: Numeric code of office location (e.g., 1749)
- race_desc: Race/ethnicity (e.g., Black, White, Asian)
- marital_desc: Marital status (e.g., Married, Single)
- performance_score: Performance evaluation label (e.g., Fully Meets, Exceeds)
- current_employee_rating: Numeric rating (e.g., 3)
- created_at: Record creation timestamp
- updated_at: Record update timestamp
- deleted_at: Record deletion timestamp (NULL if active)

### Example row:
| emp_id | first_name | last_name | title            | supervisor       | business_unit | employee_status | employee_type | pay_zone | department_type       | division  | state | gender_code | performance_score | current_employee_rating |
|--------:|-------------|-----------|------------------|------------------|----------------|------------------|----------------|-----------|------------------------|-----------|--------|--------------|------------------|--------------------------|
| 1001    | Susan       | Exantus   | Software Engineer| Angela Carlson   | BPC            | Active           | Part-Time      | Zone A    | Software Engineering   | Engineers | MA     | Female       | Fully Meets       | 3                        |

### Instruction:
Generate a valid SQL query that answers the following question, based on the schema above.
Return **only** the SQL query, without any explanation.

### Question:
{user_prompt}

### SQL Query:
"""


    try:
        output = llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.2,      # lebih stabil untuk query SQL
            top_p=0.9,
            stop=["#", ";", "User:", "\n\n"]
        )

        text = output["choices"][0]["text"].strip()

        # Hapus karakter aneh / sisa token
        clean_text = text.replace("</s>", "").replace("<s>", "").strip()

        return {"response": clean_text}

    except Exception as e:
        return {"error": str(e)}


# =======================
# 3. JALANKAN SERVER
# =======================
if __name__ == "__main__":
    # host=0.0.0.0 agar bisa diakses dari komputer lain di jaringan yang sama
    uvicorn.run(app, host="0.0.0.0", port=8000)
