import os
from openai import OpenAI

# 1. SET YOUR KEY HERE OR IN ENV VAR: export NVIDIA_API_KEY="nvapi_..."
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "PASTE_YOUR_KEY_HERE_IF_NOT_IN_ENV")

# 2. NVIDIA NIM Endpoint (Using Nemotron 3 Ultra or Llama 3.1 70B via NVIDIA hosted)
# Base URL for NVIDIA's OpenAI-compatible API
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Model name (check NVIDIA catalog for latest available names)
# Good general purpose: "meta/llama-3.1-70b-instruct" or "nvidia/nemotron-3-ultra"
LLM_MODEL = "meta/llama-3.1-70b-instruct" 

def call_llm(prompt: str, system_prompt: str = "You are an expert AML Data Engineer.", temp=0.2) -> str:
    """Simple wrapper to call NVIDIA LLM."""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return ""
