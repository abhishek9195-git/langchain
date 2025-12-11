from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model_name = 'gemini-2.5-flash-lite'
model = ChatGoogleGenerativeAI(model=model_name)

result = model.invoke('What is the capital of India ?')

print(result.content)