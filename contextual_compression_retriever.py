from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import LLMChainExtractor

from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(
        page_content=(
            "The capital of France is Paris. It is known for its art and fashion."
            "The Eiffel Tower is a famous landmark there. The weather today is sunny."
        ),
        metadata={"source": "document_1.txt"},
    ),
    Document(
        page_content=(
            "The New York Yankees won the World Series in 2009. Baseball is a popular sport."
            "The main point is about New York City's baseball history, not Paris."
        ),
        metadata={"source": "document_2.txt"},
    ),
    Document(
        page_content=(
            "Socrates was a classical Greek philosopher. He is famous for his method of inquiry called "
            "the Socratic method. He lived in Athens and was executed in 399 BC. "
            "His student was Plato."
        ),
        metadata={"source": "document_3.txt"},
    ),
    Document(
        page_content=(
            "The stock market experienced a significant surge in Q3. The CEO made a statement about profits."
            "This information is highly sensitive and financial in nature."
        ),
        metadata={"source": "document_4.txt"},
    ),
]

llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task='text-generation')
model = ChatHuggingFace(llm=llm)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

base_retriever = vector_store.as_retriever(
    search_kwargs = {'k': 3, 'lambda_mult': 1} 
)

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

query = 'Tell me something about the sport.'
compressed_result = compression_retriever.invoke(query)