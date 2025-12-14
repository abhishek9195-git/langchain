# Used for conditional chain

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableBranch, RunnablePassthrough

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
    Generate a report on topic with more than 500 words. topic is {topic}
    """,
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="""
    Summarize the report {text} in less than 100 words 
    """,
    input_variables=['text']
)

# ===================================== chain

parser = StrOutputParser()
report_generation_chain = RunnableSequence(prompt_1, model, parser)

condition_check_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200, RunnableSequence(prompt_2, model, parser)),
    RunnablePassthrough()
)

# ====================================== Invoke

final_chain = RunnableSequence(report_generation_chain, condition_check_chain)

result = final_chain.invoke('Harry Potter')

print('==> result', result)