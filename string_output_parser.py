from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    # repo_id='meta-llama/Llama-3.1-8B-Instruct',
    repo_id="google/gemma-2-2b-it",
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

# =============================================== Prompt 1

template1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables = ['topic']
)

prompt1 = template1.invoke({'topic': 'Harry potter'})
result1 = model.invoke(prompt1)

print('====> prompt1_result', result1) 

# =============================================== Prompt 2

template2 = PromptTemplate(
    template = 'Write a 5 line summary on the following text. /n {text}',
    input_variables = ['text']
)

prompt2 = template2.invoke({'text': result1.content})
result2 = model.invoke(prompt2) 
# Returns a string output. Which is an unstructured output. 
# We will use strOutputParser to convert it into structured output.

print('====> prompt2_result', result2) 

# ================================================ Form a chain with parser as bridge

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser 
# 1. Parser extracts string output. 
# 2. Extract string output from first model and send it to next model

parsed_result = chain.invoke({'topic': 'Harry Potter'})
print('====> parsed_result ', parsed_result)