from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm = llm)

schema = [
    {'name': 'fact 1', 'description': 'Fact 1 description.'},
    {'name': 'fact 2', 'description': 'Fact 2 description.'},
    {'name': 'fact 3', 'description': 'Fact 3 description.'},
    {'name': 'fact 4', 'description': 'Fact 4 description.'},
    {'name': 'fact 5', 'description': 'Fact 5 description.'}
]

parser = StrOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about the topic {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic': 'Peaky Blinders'})
print(f'====> result {result}')

