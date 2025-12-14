# mmr: Maximum marginal : Relevant to the query and diverse content. Non redundant results.
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings

docs = [
    Document(page_content='Langchain makes it easy to work with LLMs'),
    Document(page_content='Langchain is used to build LLM based applications.'),
    Document(page_content='Chroma is used to store and search document embeddings.'),
    Document(page_content='Embeddings are vector representation of a text'),
    Document(page_content='MMR helps to get diverse results when doing similarity search.'),
    Document(page_content='Langchain supports Chroma, FIASS, Pinecone and more.')
]

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs = {'k': 3, 'lambda_mult': 1} 
)
# lambda_mult = relevance-diversity balance. Zero = maximum diverse, One = Minimum diverse
# k = top results

query = 'What is langchain'
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f'\n =====> Result {i + 1}')
    print(doc.page_content)