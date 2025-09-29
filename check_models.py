import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load the GOOGLE_API_KEY from your .env file
load_dotenv()

# Configure the SDK with your API key
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    print("Successfully configured with API key.")
except Exception as e:
    print(f"Error configuring API key: {e}")
    exit()

print("\n--- Available Models ---")

# List all available models
for model in genai.list_models():
    # We only care about models that support 'generateContent' (for chat/text)
    if 'generateContent' in model.supported_generation_methods:
        print(f"Model name: {model.name}")
        # print(f"  - Display name: {model.display_name}")
        # print(f"  - Description: {model.description[:60]}...")
        # print("-" * 20)