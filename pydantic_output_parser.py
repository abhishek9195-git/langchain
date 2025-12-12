from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
)
model = ChatHuggingFace(llm = llm)

class Product(BaseModel):
    title: str = Field(description = 'Title of the product')
    description: str = Field(description='Description of the product')
    price: float = Field(gt=0, description='Price of the product')
    brand: str = Field(description='Brand of the product.')
    category: str = Field(description='Category of the product.')

parser = PydanticOutputParser(pydantic_object=Product)

template = PromptTemplate(
    template='Generate the title, description, price and brand of a product with category {category} \n {format_instruction}',
    input_variables=['category'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

#  ================================================ Invoke type 1

# prompt = template.invoke({'category': 'Electronics'})
# result = model.invoke(prompt)
# parsed_result = parser.parse(result.content)

# print(f'===> parsed_result {parsed_result}')

#  ================================================ Invoke type 2

chain = template | model | parser
result = chain.invoke({'category': 'sports'})
print('===> result title:', result.title)

