from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import torch

long_text = """
The quick brown fox jumps over the lazy dog. 
This is a simple English sentence used for testing various systems. 
It contains all letters of the alphabet, which is why it is often used as a pangram. 
The dog did not seem bothered by the fox's action.
The weather forecast predicts rain tomorrow.
We should probably bring an umbrella to the park.
The park is a great place for a walk in sunny weather.
It is important to check the weather before going outside.
"""

# 1. Load an embedding model (e.g., all-MiniLM-L6-v2)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Initialize the SemanticChunker with the embedding model
# You can adjust 'breakpoint_threshold_type' to percentile, standard_deviation, or interquartile
text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile" 
)

# 3. Create documents using the semantic splitting logic
docs = text_splitter.create_documents([long_text])

print(f"Original text has {long_text.count('.') + long_text.count('?') + long_text.count('!')} sentences.")
print(f"Number of semantic chunks created: {len(docs)}\n")

for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
