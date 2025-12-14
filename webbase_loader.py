from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
)
model = ChatHuggingFace(llm = llm)

url = 'https://docs.langchain.com/oss/javascript/langchain/overview'
loader = WebBaseLoader(url)
docs = loader.load()

prompt = PromptTemplate(
    template="""
    Answer the following question \n {question} from the following text \n {text}
    """,
    input_variables=['question', 'text']
)

parser = StrOutputParser()

try:
    chain = prompt | model | parser

    result = chain.invoke({
        'question': 'What is this information about ?',
        'text': docs[0].page_content
    })
    print('=======> result', result)
    
except FileNotFoundError:
    print('Directory not found')
except Exception as e:
    print('Error {e}')