from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_google_gemini_tools_agent
from langchainhub import hub
# from langchain import hub
from langchain_core.tools import tool

from dotenv import load_dotenv
import requests

load_dotenv()

# ================================== model

# model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

search_tool = DuckDuckGoSearchRun()

# ================================== model

# prompt = hub.pull('hwchase17/react')

# ================================== Create agent

# agent = create_agent(
#     model,
#     tools=[search_tool],
#     # prompt = prompt
#     system_prompt='You are a helpful assistant. Be concise and accurate.',
    
# )

# agent = create_google_gemini_tools_agent()

# agent_executor = AgentExecutor(
#     agent = agent,
#     tool = [search_tool],
#     verbose = True
# )
prompt = hub.pull("hwchase17/google-structured-chat")

agent = create_google_gemini_tools_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True # Set verbose=True to see the internal loop in action
)

response = agent_executor.invoke({"input": 'Today news for india.'})

print("\n--- AGENT EXECUTOR CREATED ---")
print(f"The AgentExecutor manages the loop and runs the tools.")

print('==> response', response)