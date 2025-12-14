from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('assets/resume.pdf')
docs = loader.load() # Per page one doc

text_splitter = CharacterTextSplitter(
    separator="\n",        # Try splitting on new lines first
    chunk_size=200,        # Max size of each chunk in characters
    chunk_overlap=20,      # Number of characters to overlap between chunks
    length_function=len,   # The function used to measure length (defaults to len)
    is_separator_regex=False,
)

documents = text_splitter.split_documents(docs)

for i, doc_chunk in enumerate(documents):
    print(doc_chunk.page_content, '\n')
