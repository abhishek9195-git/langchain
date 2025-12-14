from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

# Create document

documents = [
    Document(page_content='LangChain helps developers build LLM applications.'),
    Document(page_content='chroma is a vector database optimized for LLM based search.'),
    Document(page_content='Embeddings convert text into high-dimensional vectors.'),
    Document(page_content='Open AI provides powerful embedding models.')
]

# Embedding model

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create chroma vector store in memory

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name='my_collection'
)

# Convert vector store into a retriever

retriever = vector_store.as_retriever() # k = number of relevant results.
retriever.search_kwargs['k'] = 2

query = 'What is chroma used for ?'
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f'\n =====> Result {i + 1}')
    print(doc.page_content)