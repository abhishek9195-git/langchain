from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# ====================================== model declarations

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

# ====================================== Pydantic class output parser

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Provide the sentiment of the feedback.')

pydantic_output_parser = PydanticOutputParser(pydantic_object=Feedback)

# ====================================== Prompt declaration with input variable

prompt_1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text positive or negative \n {feedback} {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': pydantic_output_parser.get_format_instructions()}
)

prompt_2 = PromptTemplate( # Positive response
    template="""
    Respond to this positive feedback \n {feedback} with General and empathetic tone.
    Use the below format to respond.
    
    Empathetic and Action-Oriented tone

    Thank you for bringing this to our attention. We're sorry to hear you're experiencing issues with your smartphone hanging. This is certainly not the experience we aim for. We'd like to help resolve this. 
    Could you please contact our support team at [xyz@support.xyz.com or 1800 848 48484] so we can troubleshoot this with you and find a solution?

    """,
    input_variables=['feedback']
)

prompt_3 = PromptTemplate( # Negative response
    template="""
    Respond to this negative feedback \n {feedback}

    Example:

    Thank you for taking the time to share your feedback. We are glad that you are happy.
    """,
    input_variables=['feedback']
)

# ====================================== Declare classifier chain

# classifies the feedback into {sentiment: 'positive'} or {sentiment: 'negative'}
classifier_chain = prompt_1 | model | pydantic_output_parser

# ====================================== Declare Conditional chain: RunnableBranch

string_output_parser = StrOutputParser()

# branch_chain = RunnableBranch(
#     # Access the nested value using dictionary keys
#     (lambda x: x['sentiment_result'].sentiment == 'positive', prompt_2 | model | string_output_parser),
#     (lambda x: x['sentiment_result'].sentiment == 'negative', prompt_3 | model | string_output_parser),
#     RunnableLambda(lambda x: 'Could not find sentiment')
# )

branch_chain = RunnableBranch(
    # Access the nested value using dictionary keys
    (lambda x: x['sentiment'] == 'positive', prompt_2 | model | string_output_parser),
    (lambda x: x['sentiment'] == 'negative', prompt_3 | model | string_output_parser),
    RunnableLambda(lambda x: 'Could not find sentiment')
)
# ====================================== Merge both chains

final_chain = classifier_chain | branch_chain

# final_chain = RunnableParallel(
#     sentiment_result=classifier_chain,
#     feedback=RunnableLambda(lambda x: x['feedback'])
# ) | branch_chain

# ====================================== Invoke chain

result = final_chain.invoke({'feedback': 'This smartphone hangs a lot.'})

print('==> result', result)
