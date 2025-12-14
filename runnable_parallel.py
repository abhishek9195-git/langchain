from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnableSequence

from dotenv import load_dotenv

load_dotenv()

# ====================================== Model

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct'
)

model = ChatHuggingFace(llm=llm)
# model = ChatGoogleGenerativeAI(
#     model='gemini-2.5-flash-lite'
# )

# ====================================== Prompt

prompt_1 = PromptTemplate(
    template="""
    Generate 5 male characters name list of the tv-show {topic}
    """,
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="""
    Generate 5 female characters name list of the tv-show {topic}
    """,
    input_variables=['topic']
)

# ======================================= Parallel chain

parser = StrOutputParser()

parallel_chain = RunnableParallel({ # Returns dictionary response
    'male_characters': RunnableSequence(prompt_1, model, parser),
    'female_characters': RunnableSequence(prompt_2, model, parser)
})

result = parallel_chain.invoke({'topic': 'Harry Potter'})
print('==> result', result)