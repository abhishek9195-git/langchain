from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 
# Create chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'), # Load previous context/ conversation
    ('human', '{query}') # Query is the current message which human asked.
])

# Load chat history
chat_history = []
with open('chat_history.txt') as file:
    for line in file.readlines():
            # Use eval() to convert the string representation back into the LangChain object
            # This works because HumanMessage and AIMessage are defined and imported
            try:
                message_object = eval(line.strip())
                chat_history.append(message_object)
            except Exception as e:
                print(f"Could not parse line: {line.strip()} - Error: {e}")

print(f"====> chat_history {chat_history}")

# Create the prompt and inject the history to MessagePlaceholder
# We are sending two placeholders:- 1. chat_history, 2. query that user is asking.
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund ?'})

print(f"====> prompt {prompt}")
