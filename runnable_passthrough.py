# Used to get the output of previous sequential chain into the current chain 
# returns input as output, without any changes.

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough

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
    Generate 5 male characters name list of the tv-show {tv_show_name}
    """,
    input_variables=['tv_show_name']
)

prompt_2 = PromptTemplate(
    template="""
    Generate 3 intersting fact about each characters in {male_character_list}
    """,
    input_variables=['male_character_list']
)

# ======================================= Chain

parser = StrOutputParser()
chain_actor_list = RunnableSequence(prompt_1, model, parser)

chain_actor_facts = RunnableParallel({
    'actor_list': RunnablePassthrough(),     # returns result of previous chain into current chain
    'facts': RunnableSequence(prompt_2, model, parser)
})

final_chain = RunnableSequence(chain_actor_list, chain_actor_facts)

# ======================================= Invoke

result = final_chain.invoke({'tv_show_name': 'Stranger things'})
print(result)