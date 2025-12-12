from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-lite'
)

messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about langchain in few words.')
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)

# Run: python messages.py