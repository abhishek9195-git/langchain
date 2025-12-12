from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate(# Creates dynamic multiturn templates
    [
        ('system', 'You are a helpful {domain} expert'),
        ('human', 'Explain in simple terms, What is the {topic}')
    ]
)

prompt = chat_template.invoke(
    {
    'domain': 'Cricket',
    'topic': 'Run out'
    }
)

print(prompt)