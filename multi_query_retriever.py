from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langchain_community.retrievers import MultiQueryRetriever
from langchain_classic.retrievers import MultiQueryRetriever

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task='text-generation')
model = ChatHuggingFace(llm=llm)

# ====================== Create vector store

documents = [
    Document(page_content="Exercise improves cardiovascular health and longevity."),
    Document(page_content="A healthy diet reduces the risk of chronic diseases like diabetes and heart issues."),
    Document(page_content="Meditation helps reduce stress and anxiety, improving mental clarity."),
    Document(page_content="Regular sleep patterns are crucial for physical recovery and cognitive function."),
]

embedding_model = HuggingFaceEmbeddings()

vector_store = FAISS.from_documents(
    documents=documents, 
    embedding=embedding_model
)

# ======================== Retriever

base_retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs = {'k': 3, 'lambda_mult': 1} 
)


multi_query_retriever = MultiQueryRetriever.from_llm(
    base_retriever=base_retriever,
    llm = model
)

original_question = "What are the health benefits of both exercise and a good diet?"

print(f"Original Query: {original_question}\n")
print("--- Generated alternative queries will be logged below ---")

results = multi_query_retriever.invoke(original_question)

print("\n--- Unique Retrieved Documents ---")
for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content}")
