'''
WHAT THIS PATTERN IS:
Pydantic enforces schema validation on model output at generation time. A Pydantic BaseModel which encompasses the schema is passed along with the
input to the LLM call. The llm output should conform to this schema.

WHEN YOU'D USE THIS:
When we need structured outputs from LLM models in our pipelines. Especially when downstream application code read model responses and processes
them, we want to enforce a proper schema so any logic can process the objects unambiguously. For example a model that reads tickets submitted
to our customer support portal to output Python objects containing keys like 'priority', 'summary', 'category' in a batch processing job

WHAT PROBLEM IT SOLVES:
All problems caused due to model being inconsistently right (right 99% of the time). But schema inconsistencies like missing required
keywords, wrong keywords, type mismatches, hallucinated extra details or malformed JSON responses that could corrupt application state silently 
and wreak havoc when downstream services iterate through keys of the faulty objects causing KeyErrors, Type errors making it difficult to trace the root cause

THE FLOW:
input → ChatPromptTemplate (with format instructions) → LLM → PydanticOutputParser → validated Python object

CONSTRUCTS INTRODUCED TODAY:
PydanticOutputParser (which will replace our StrOutputParser)
parser.get_format_instructions() — this is the key thing that's different from Day 1. It injects schema instructions into your prompt so the model knows what structure to output.
Pydantic BaseModel class
'''

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, message
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal

# Field() used to customize BaseModel attributes. Set default values and constraints.

# Set additional properties for the fields using JSON Schema metadata
# JSON Schema metadata: main ones are title (like a variable name), description, examples of the fields

llm = ChatOpenAI(model='gpt-4o-mini')

class SupportTicket(BaseModel):
    # ticket_id: int = Field(default_factory=lambda: uuid.uuid4())
    priority: Literal['low', 'medium', 'high'] = Field(description='The priority assigned for the issue. Can be one of low, medium, high')
    summary: str = Field(description='The summary of the support ticket')
    category: Literal['General', 'Technical', 'Billing'] = Field(description='The category of the support ticket. Can be one of General, Billing, Technical')


messages = [
    ('system', 'You are a customer support ticket processing chatbot that reads content of customer support tickets and assigns priority, a crisp message summary and category of the issue. \n {format_instructions}'),
    ('human', '{ticket_text}'),
]

parser = PydanticOutputParser(pydantic_object=SupportTicket)

prompt_template = ChatPromptTemplate(
    messages=messages,
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = prompt_template | llm | parser
response = chain.invoke({
    'ticket_text': 'I was charged twice for my iPad purchase. Please arrange a refund!!'
})

print(type(response))
print(response)

# POST ANALYSIS BLOCK:

# PydanticOutputParser: Define output schema for a model and also validate the output against this schema. Expects Pydantic BaseModel object defining
# the fields, types, optional default values and constraints and give metadata for the JSON schema.

# get_format_instructions(): Method that is used as a partial variable in a prompt template so we can stuff the output requirements in the prompt.
# partial_variables: Pre-fills prompt template variables at creation time (not at invoke time).
# Use when a variable is constant across all invocations — like format_instructions.

# Field: Individual field function that allows you to define each field (data type, default values, constraints and metadata) in your schema.

# Literal: When you want output to be assigned from a list of values (categories, classes you are trying to assign with a model)

# One sentence explanation: PydanticOutputParser helps enforce robust output contract to be followed by an LLM. Useful when model outputs are
# used by downstream applications and systems and expect the output to be of a certain shape to handle correctly