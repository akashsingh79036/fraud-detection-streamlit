import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. ALWAYS load the environment variables first!
load_dotenv()

# 2. Now pull the key from your environment variables
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# 3. Now initialize the client with the valid key
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

# 4. Define your model name
LLM_MODEL = "meta/llama-3.1-70b-instruct"


# 5. Your function stays exactly the same
def call_llm(prompt: str, system_prompt: str = "You are an expert AI Data Engineer.", temp=0.2) -> str:
    """Simple wrapper to call NVIDIA LLM."""
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
        return ""


# 6. Test block at the bottom
if __name__ == "__main__":
    print("Sending prompt to NVIDIA...")
    test_response = call_llm(prompt="Say hello world!")
    print("\nResponse:")
    print(test_response)
