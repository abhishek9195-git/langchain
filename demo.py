from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
paper_input = 'Attention is all you need.'
style_input = 'Beginner-friendly'
length_input = 'Short (1) paragraphs'

template = load_prompt('template.json')

#=================== Way 1:- 

# # Fill the prompt
# prompt = template.invoke({
#     'paper_input': paper_input,
#     'style_input': style_input,
#     'length_input': length_input
# })

# # Invoke the model
# result = model.invoke(prompt)


#==================== Way 2

template = load_prompt('template.json')
chain = template | model

result = chain.invoke({
    'length_input': length_input,
    'paper_input': paper_input,
    'style_input': style_input,
})

print(result.content)
