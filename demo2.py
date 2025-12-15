from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

search_tool = DuckDuckGoSearchRun()

agent = create_agent(
    model=model,
    system_prompt='You are a helpful assistant.',
    tools=[search_tool]

)
response = agent.invoke({'messages': ['Todays news for india.']})


