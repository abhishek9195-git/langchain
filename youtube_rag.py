# Load video transcript [Using YTLoader or Youtube API]

# ================================= Step 1: Indexing

# Text splitter: Convert transcript into multiple chunks
# Generate Embedding of all chunks and store it into vector store.

# ================================= Step 2: Retrieving [Semantic Search (Retriever)]

# Send query to retriever. Retriever will to perform semantic search. Bring relevant context.

# ================================= Step 3: Augmentation [Prompt (Context + Query)]

# Prompt creation

# ================================= Step 4: Generation

# Prompt => LLM ==> Response

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

video_id = 'Gfr50f6ZBvo'
transcript = None
preserve_formatting = False
languages = ("en",)
try:

# =================================== Download transcript

    transcript_data = YouTubeTranscriptApi().list(video_id).find_transcript(languages).fetch(preserve_formatting=preserve_formatting)

    transcript = ' '.join(chunk.text for chunk in transcript_data)
    print('==> transcript length:', len(transcript))

# =================================== Split into chunks

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # chunks = text_splitter.split_text(transcript)
    chunks = text_splitter.create_documents([transcript]) 


    print(f'Total no. of chunks', len(chunks))

# =================================== Convert chunks into vectors

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    # print('==> Chunk vector ids', vector_store.index_to_docstore_id)
    # print(vector_store.get_by_ids(['chunk_id']))


# =================================== Retriever

# Finds the vector in vector store which is closest to the query vector. Returns chunk.
# Input: A Query, Output: List of Document

    retriever = vector_store.as_retriever(
        search_type = 'mmr',
        search_kwargs = {'k': 2, 'lambda_mult': 1} 
    )
    # lambda_mult = relevance-diversity balance. Zero = maximum diverse, One = Minimum diverse
    # k = top results

    # query = 'What is deepmind'
    # results = retriever.invoke(query)

    # for i, doc in enumerate(results):
    #     print(f'\n =====> Result {i + 1}')
    #     print(doc.page_content)


# =================================== Prompt creation

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, Just say that you don't know.

        {context}
        question: {question}
        """,
        input_variables=['context', 'question']
    )


    question = 'Is the topic aliens discussed in this vide ? If yes, then what was discussed ?'
    retrieved_document_list = retriever.invoke(question)

    context_text = '\n \n'.join(doc.page_content for doc in retrieved_document_list)

    formatted_prompt = prompt.invoke({'context': context_text, 'question': question})

# =================================== LLM Model invoke

    llm = HuggingFaceEndpoint(
        repo_id='meta-llama/Llama-3.1-8B-Instruct',
        task='text-generation'
    )

    model = ChatHuggingFace(llm = llm)

    # parser = StrOutputParser()

    # chain = formatted_prompt | model | parser
    
    answer = model.invoke(formatted_prompt)
    print('==> result', answer.content)



except TranscriptsDisabled:
    print('No caption available for this video.')

except Exception as e:
    print(e)