import os
from dotenv import load_dotenv
import subprocess
import sys
import streamlit as st

# Hot-patch setup to handle missing library bugs automatically
try:
    from openai import OpenAI
except ImportError:
    print("⚠️ OpenAI package missing on host. Executing live hot-patch install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai>=1.0.0"])
    from openai import OpenAI
    
# 1. ALWAYS load the environment variables first!
load_dotenv()

# 2. Now pull the key from your environment variables
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# --- STREAMLIT CLOUD FALLBACK SECURE SEARCH ENGINE ---
if not NVIDIA_API_KEY:
    try:
        # Fall back to Streamlit Cloud's internal secure vault if local .env is missing
        NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]
    except Exception:
        NVIDIA_API_KEY = None
# ----------------------------------------------------

# 3. Now initialize the client safely if a key is available to prevent startup crashes
if NVIDIA_API_KEY:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    )
else:
    client = None

# 4. Define your model name
LLM_MODEL = "meta/llama-3.1-70b-instruct"


# 5. Your function stays exactly the same, but with a built-in safety check
def call_llm(prompt: str, system_prompt: str = "You are an expert AI Data Engineer.", temp=0.2) -> str:
    """Simple wrapper to call NVIDIA LLM."""
    if not client:
        return "❌ LLM Error: Missing API Credentials. Please set your NVIDIA_API_KEY variable."
        
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return f"❌ LLM Error: {e}"


# 6. Test block at the bottom
if __name__ == "__main__":
    print("Sending prompt to NVIDIA...")
    test_response = call_llm(prompt="Say hello world!")
    print("\nResponse:")
    print(test_response)
