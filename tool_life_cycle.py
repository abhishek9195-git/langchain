from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv

load_dotenv()

# ====================================== Model creation

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

result = llm_with_tools.invoke('Can you multiply 3 and 5 ?')

# print(result.content)
# print(result.tool_calls[0])

# ====================================== Tool Execution

# ToolMessage: When we execute a tool with the help of a tool call

tool_message = multiply.invoke(result.tool_calls[0])

print(tool_message)