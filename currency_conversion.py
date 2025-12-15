from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from langchain_core.tools import InjectedToolArg
from typing import Annotated
from dotenv import load_dotenv
import json
from langchain_core.messages import HumanMessage, ToolMessage


load_dotenv()

# ====================================== Model creation

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# ====================================== Tool creation

@tool
def fetch_conversion_rate(base_currency: str, target_currency: str) -> float:
    """
    This function fetches the currency conversion rate between a given base currency and target currency.
    """
    url = f'https://v6.exchangerate-api.com/v6/ae180478acd133d4e7c88d89/pair/{base_currency}/{target_currency}'

    response = requests.get(url)

    return response.json()

# InjectedToolArg: Because llm won't call utilize result of fetch_conversion_rate function to set conversion_rate, 
# We will tell llm to ignore this argument and don't set any value. Developer will set this. .
@tool 
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    Given a currency conversion rate this function calculates the target currency value from a given base currency value
    """
    return base_currency_value * conversion_rate


# tool_response = get_conversion_factor.invoke({'base_currency': 'USD', 'target_currency': 'INR'})
# print(tool_response.conversion_rate)


# ====================================== Tool binding with llm

llm_with_tools = model.bind_tools([fetch_conversion_rate, convert])

# ====================================== LLM invoke

query = HumanMessage('What is the conversion rate between USD and INR, and based on that can you convert 10 USD to INR ?')
messages = [query]

# --- First LLM Call (Generates Tool Calls) ---
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

# print(ai_message.tool_calls) # Should show two tool calls

conversion_rate = None # Initialize rate

# --- Process Tool Calls ---
for tool_call in ai_message.tool_calls:
    # Use dot notation for tool_call attributes
    tool_name = tool_call['name'] 
    tool_args = tool_call['args']
    
    if tool_name == 'fetch_conversion_rate':
        # Invoke the tool using ONLY the arguments dictionary
        tool_raw_result = fetch_conversion_rate.invoke(tool_args)
        print('==> tool_raw_result', tool_raw_result)
        # tool_raw_result is likely a dictionary like {'conversion_rate': 83.0}
        conversion_rate = tool_raw_result['conversion_rate']

        # Append the result as a proper ToolMessage to the history
        messages.append(ToolMessage(content=json.dumps(tool_raw_result), tool_call_id=tool_call['id']))

    elif tool_name == 'convert':
        # Add the dynamically found conversion rate to the arguments
        tool_args['conversion_rate'] = conversion_rate
        
        # Invoke the convert tool
        tool_raw_result = convert.invoke(tool_args)
        
        # tool_raw_result might be just a float or int
        print(f"Tool 'convert' raw output: {tool_raw_result}")

        # Append the result as a proper ToolMessage to the history
        messages.append(ToolMessage(content=str(tool_raw_result), tool_call_id=tool_call['id']))


# print('==> messages', messages) # Verify message history looks correct

# --- Second LLM Call (Generates Final Answer) ---
# The model should now have all the info (the tool results) to formulate a response
llm_response = llm_with_tools.invoke(messages)

print('==> llm llm_response content:', llm_response.content)

