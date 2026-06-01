import requests

# URL API kamu dari LocalTunnel
API_URL = "https://llamaapi.loca.lt/generate"

AUTH_TOKEN = None  # contoh: "Bearer rahasia123"

def chat_with_llama():
    print("=== Chat dengan LLaMA Lokal ===")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        prompt = input("Anda: ")
        if prompt.lower() in ["exit", "quit", "keluar"]:
            print("Keluar dari chat.")
            break

        payload = {"prompt": prompt, "max_tokens": 200}

        headers = {"Content-Type": "application/json"}
        if AUTH_TOKEN:
            headers["Authorization"] = AUTH_TOKEN

        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            if response.status_code == 200:
                data = response.json()
                print(f"LLaMA: {data.get('response', '(tidak ada respons)')}\n")
            else:
                print(f"[Error] Status Code: {response.status_code} | {response.text}")
        except Exception as e:
            print(f"[Exception] {e}")

if __name__ == "__main__":
    chat_with_llama()
