# Used to provide a custom logic in chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda

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

# ====================================== Runnable Lambda function for word_count

def word_count(text: str):
    return len(text.split())

runnable_word_count_fn = RunnableLambda(word_count)
# print(runnable_word_count_fn.invoke('Tommy Shelby'))

# ====================================== Prompt

prompt_1 = PromptTemplate(
    template="""
    Generate 1 male character name of the tv-show {tv_show_name}

    Example:
    John Doe

    DO NOT ADD any description to the name. Generate only the name.
    """,
    input_variables=['tv_show_name']
)

# prompt_2 = PromptTemplate(
#     template="""
#     Generate 3 intersting fact about each characters in {male_character_list}
#     """,
#     input_variables=['male_character_list']
# )

# ======================================= Chain

parser = StrOutputParser()
chain_actor_name = RunnableSequence(prompt_1, model, parser) # Tommy Shelby

chain_actor_name_and_word_count = RunnableParallel({
    'actor_name': RunnablePassthrough(), # returns input as output, without any changes.
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(chain_actor_name, chain_actor_name_and_word_count)

# ======================================= Invoke

result = final_chain.invoke({'tv_show_name': 'Stranger things'})
print(result)