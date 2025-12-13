from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel

from dotenv import load_dotenv

load_dotenv()

# ====================================== model declarations
llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct'
)

lama_model = ChatHuggingFace(llm=llm)
google_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-lite'
)


# ======================================= Prompt declarations
prompt_1 = PromptTemplate(
    template='Provide 5 male character names of TV show {tv_show_name}',
    input_variables='tv_show_name'
)

prompt_2 = PromptTemplate(
    template='Provide 5 female chracter names of TV show {tv_show_name}',
    input_variables='tv_show_name'

)

prompt_3 = PromptTemplate(
    template='Merge the provided male character names and female character names into a single document \n tv_show_characters {male_characters} and {female_characters}',
    input_variables=['male_characters', 'female_characters']
)

# ======================================= Create parallel chains using runnable parallel

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'male_characters':  prompt_1 | lama_model | parser,
    'female_characters': prompt_2 | google_model | parser
})

sequential_chain = prompt_3 | lama_model | parser

final_chain = parallel_chain | sequential_chain

result = final_chain.invoke({'tv_show_name': 'Peaky Blinders'})
print('===> result', result)
