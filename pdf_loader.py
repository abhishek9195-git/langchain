from langchain_community.document_loaders import PyPDFLoader

# 1. Initialize the loader with the path to your PDF file
file_path = "Payment_Receipt.pdf" 
loader = PyPDFLoader(file_path)

# 2. Load the documents
# This returns a list of Document objects
documents = loader.load()

# Optional: Load and split documents in one step
# pages = loader.load_and_split()

# 3. Process the loaded documents (e.g., print content of the first page)
print(f"Loaded {len(documents)} pages.")
if documents:
    print(f"Content of the first page:\n{documents[0].page_content[:500]}...")
    print(f"Metadata of the first page:\n{documents[0].metadata}")
