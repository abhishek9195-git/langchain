from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

# ==================================================== Invoke type 1
# messages = [
#     ("system", "You are a helpful assistant."),
#     ("human", "Hi"),
# ]
# result = model.invoke(messages)

# ==================================================== Invoke type 2

# template = PromptTemplate(

#     template='What is the capital of {country_name}?',
#     input_variables='country_name'
# )
# prompt = template.invoke({'country_name': 'France'})
# result = model.invoke(prompt)
# print(result)

# ==================================================== Invoke type 3

# template = PromptTemplate(

#     template='What is the capital of {country_name}?',
#     input_variables='country_name'
# )

# chain = template | model
# result = chain.invoke({'country_name': 'Germany'})
# print(result)


# ==================================================== Invoke with parser

parser = StrOutputParser()
template = PromptTemplate(

    template='What is the capital of {country_name}?',
    input_variables='country_name'
)

chain = template | model | parser
result = chain.invoke({'country_name': 'Germany'})
print(result)