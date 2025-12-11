from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv

import os

load_dotenv()

os.environ['HF_HOME'] = 'D:/hf_cache'
# Load the model directly to your server's memory
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-large")

# Wrap it into a LangChain LLM
model = HuggingFacePipeline(pipeline=hf_pipeline)

# Use it as a standard LangChain component
print(model.invoke("Translate English to French: I love AI"))

