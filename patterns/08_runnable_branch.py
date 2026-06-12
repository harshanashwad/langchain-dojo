"""
PATTERN: RunnableBranch
DAY: 8

PURPOSE: To use conditional logic in a chain
USE WHEN: When you want to execute a specific branch based on a particular condition (For example, category of tickets deciding which chain will be executed for that ticket)
FLOW: 
input_dict
  → RunnableBranch(
      (condition_fn_1, chain_1), condition_fn_1 could be a function/callable that returns a boolean
      (condition_fn_2, chain_2),
      default_chain        ← no condition, always runs if nothing matches
    )
  → output of whichever chain ran

NEW CONSTRUCTS: 
RunnableBranch
"""


from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
from langchain_openai import ChatOpenAI
from typing import Literal


#  BoilerPlate to understand RunnableBranch

raw_ticket_text = 'What kind of products are there in the platform other than electronics?'

def sanitize(input_dict):
  return {
    **input_dict,
    'raw_ticket_text': input_dict['raw_ticket_text'].strip().capitalize()
  }

llm = ChatOpenAI(model='gpt-4o-mini')

class SupportTicket(BaseModel):
  ticket_summary: str = Field(description='Concise summary of the customer issue in the ticket')
  category: Literal['General', 'Billing', 'Technical'] = Field(description='The category of the customer support ticket')
  priority: Literal['high', 'medium', 'low'] = Field(description='Appropriate priority level for the issue')
  
parser = PydanticOutputParser(pydantic_object=SupportTicket)
string_parser = StrOutputParser()

cpt = ChatPromptTemplate(
  messages=[
    ('system', 'You are a customer support ticket assistant. Analyze the raw ticket text delimited by triple backticks and assign a category (Billing, Technical, General), priority (low, medium or high) and a crisp summary of the customer issue \n {format_instructions}'),
    ('human', '{raw_ticket_text}'),
  ],
  partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = RunnableLambda(sanitize) | cpt | llm | parser

ticket = chain.invoke({
  'raw_ticket_text': raw_ticket_text
})

print(ticket)
print('––––––––––––––––––––––')

# Prompts based on category
billing_prompt = "You are a billing specialist. Address this ticket focusing on refunds and payment resolution: {ticket_summary}"
technical_prompt = "You are a tech support engineer. Address this ticket with troubleshooting steps: {ticket_summary}"
general_prompt = "You are a helpful support agent. Address this ticket politely: {ticket_summary}"

billing_prompt_template = ChatPromptTemplate(messages=[
    ('user', billing_prompt)
])

technical_prompt_template = ChatPromptTemplate(messages=[
    ('user', technical_prompt)
])

general_prompt_template = ChatPromptTemplate(messages=[
    ('user', general_prompt)
])

branch_chain = RunnableBranch(
    (lambda x: x['category'] == 'Billing', billing_prompt_template | llm | string_parser),
    (lambda x: x['category'] == 'Technical', technical_prompt_template | llm | string_parser),
    general_prompt_template | llm | string_parser

)

# print(ticket.model_dump())

final_response = branch_chain.invoke(ticket.model_dump())
print(final_response)

'''
POST-ANALYSIS BLOCK

lambda function is good enough for the callable in each branch. Consider using RunnableBranch when the branching logic is a few lines of code from another function (say)

model_dump() comes in handy when you want to convert a BaseModel Class object (from pydantic parser) into a dictionary to use it downstream on another chain

RunnableBranch:
Run chain conditionally when you when to route the input to different downstream chains based on matching criteria
'''
