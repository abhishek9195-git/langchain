from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableSequence

from dotenv import load_dotenv

load_dotenv()

# ====================================== Model
llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct'
)

model = ChatHuggingFace(llm=llm)
google_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-lite'
)

# ====================================== Prompt

prompt_1 = PromptTemplate(
    template="""
    Generate 5 characters name list of the tv-show {topic}
    """,
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="""
    Generate 3 interesting fact about each character with character name provided in text - {text}
    """,
    input_variables=['text']
)
# ====================================== Sequential chain creation

parser = StrOutputParser()

chain = RunnableSequence(prompt_1, model, parser, prompt_2, model, parser)

# ====================================== Chain invoke

result = chain.invoke({'topic': 'Peaky Blinders'})

print('==> result', result)