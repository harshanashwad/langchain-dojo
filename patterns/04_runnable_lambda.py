'''
PATTERN: Runnable Lambda
DAY: 4

PURPOSE: Pattern is about chaining custom functions in an LCEL chain. RunnableLambda is a wrapper around custom functions so it can be chained together
in LCEL. Can be used to add tools to a chain that can be used by an LLM to take action

USE WHEN: LCEL only supports langchain native components to be chained. When you want custom functions to be invoked in sequence in a chain, 
use this wrapper. Best use case is tools to expose custom functionality to our LLM app like web search, running code, executing a db query etc.
Not everything requires us to invoke a model to get something

FLOW: chain = prompt template | RunnableLambda(custom_function) | llm | StrOutputParser

NEW CONSTRUCTS: 
RunnableLambda wrapper -  Wrap custom functions in an LCEL using this class
'''

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

class SupportTicket(BaseModel):
    # ticket_id: int = Field(default_factory=lambda: uuid.uuid4())
    priority: Literal['low', 'medium', 'high'] = Field(description='The priority assigned for the issue. Can be one of low, medium, high')
    summary: str = Field(description='The summary of the support ticket')
    category: Literal['General', 'Technical', 'Billing'] = Field(description='The category of the support ticket. Can be one of General, Billing, Technical')


parser = PydanticOutputParser(pydantic_object=SupportTicket)


messages = [
    ('system', 'You are a customer support ticket processing chatbot that reads content of customer support tickets and assigns priority, a crisp message summary and category of the issue. \n {format_instructions}'),
    ('human', '{ticket_text}'),
]

cpt = ChatPromptTemplate(
    messages=messages,
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

llm = ChatOpenAI(model='gpt-4o-mini')

# Custom function that strips text and capitalizes
def ticket_sanitizer(input_dict):
    return {
        **input_dict,
        'ticket_text': input_dict['ticket_text'].strip().capitalize()
    }

chain =  RunnableLambda(ticket_sanitizer) | cpt | llm | parser

response = chain.invoke({
    'ticket_text': '  please help me with my laptop. it keeps shutting down!! '
})

print(response)

'''
POST-ANALYSIS BLOCK
RunnableLambda(): Wrapper around any custom function to make it usable in a LCEL chain. Ensure when you chain a custom function. It should return
what you would use to call the next component independently using invoke(). In our case, chat prompt template expects a dict, so you return
that

This construct is cleaner to avoid modifying original dict and avoid side effects
return {
        **input_dict,
        'ticket_text': input_dict['ticket_text'].strip().capitalize()
    }
'''

