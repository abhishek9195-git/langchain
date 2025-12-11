from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()

# Switching models to verify if the infrastructure is the bottleneck
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

response = llm.invoke("What is the capital of India?")
print(response)
