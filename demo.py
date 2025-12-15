from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv

load_dotenv()

# ====================================== Model creation

llm = HuggingFaceEndpoint(
        repo_id='Qwen/Qwen-72B-Chat',
        task='text-generation'
)

# model = ChatHuggingFace(llm = llm)
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

# ====================================== Tool creation

@tool
def multiply(x: int, y: int) -> int:
    """
    Multiplies two integer numbers.

    Args: 2 integer inputs x and y.

    Returns: multiplication of x and y.
    """
    return x * y


# ====================================== Tool binding

llm_with_tools = model.bind_tools([multiply])

# ====================================== Tool calling

query = HumanMessage('Can you multiply 10 and 20 ?')
messages = [query]

ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

tool_result = multiply.invoke(ai_message.tool_calls[0])
print(tool_result) # content: 200


