from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

# Set your API token as an environment variable or pass it directly
api_token = os.getenv("HF_TOKEN")

if not api_token:
    raise ValueError("HF_TOKEN not found in environment variables or .env file.")


client = InferenceClient(token=api_token)

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Define your prompt using the chat structure
messages = [
    {"role": "system", "content": "You are a friendly assistant who always responds concisely."},
    {"role": "user", "content": "What is the capital of France?"}
]

# Use the client to generate a completion
completion = client.chat_completion(
    model=model_id,
    messages=messages,
    max_tokens=100,
)

print(completion.choices[0].message.content)
