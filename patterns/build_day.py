
'''
# Day 7: Build Day

What you're building: A support ticket pipeline that uses all 6 patterns in one chain.

```
raw ticket text
  → sanitize input          (RunnableLambda)
  → classify ticket         (ChatPromptTemplate | llm | PydanticOutputParser)
  → branch on priority      (RunnableParallel)
      - generate response   (ChatPromptTemplate | llm | StrOutputParser)
      - keep ticket data    (RunnablePassthrough)
  → final output dict
```

Rules:
- No copy-pasting from previous files — build from memory
- You can open docs only for specific forgotten syntax
- Every pattern must appear at least once

---

Before you write a single line — sketch the chain on paper or in comments first. Tell me:

1. What does your `PydanticOutputParser` schema look like?
2. Where does `RunnablePassthrough` preserve what?
3. What does the final output dict contain?
'''

'''
Chain Sketch
1. pydantic schema: category-Literal (General, Technical, Billing), priority: Literal (medium, high, low), ticket_summary, original_ticket_text cause it is needed in the runnableParallel later?
2. It preserves raw ticket text before any transformation in RunnableParallel, maybe a filed called original_ticket_text
3. response and original_ticket_text

'''
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from typing import Literal

# Two functions to wrap with RunnableLambda - sanitize and ticket_to_dict

raw_ticket_text = 'I have been charged twice on my iPhone purchase!!! I demand a refund immediately!!'
def sanitize(input_dict):
  return {
    **input_dict,
    'raw_ticket_text': input_dict['raw_ticket_text'].strip().capitalize()
  }


# Chain 1 components

llm = ChatOpenAI(model='gpt-4o-mini')

class SupportTicket(BaseModel):
  ticket_summary: str = Field(description='Concise summary of the customer issue in the ticket')
  category: Literal['General', 'Billing', 'Technical'] = Field(description='The category of the customer support ticket')
  priority: Literal['high', 'medium', 'low'] = Field(description='Appropriate priority level for the issue')
  
parser = PydanticOutputParser(pydantic_object=SupportTicket)

cpt = ChatPromptTemplate(
  messages=[
    ('system', 'You are a customer support ticket assistant. Analyze the raw ticket text delimited by triple backticks and assign a category (Billing, Technical, General), priority (low, medium or high) and a crisp summary of the customer issue \n {format_instructions}'),
    ('human', '{raw_ticket_text}'),
  ],
  partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain1 = RunnableLambda(sanitize) | cpt | llm | parser

ticket = chain1.invoke({
  'raw_ticket_text': raw_ticket_text
})

print(ticket)

# Chain 2 components

string_parser = StrOutputParser()

response_prompt = PromptTemplate.from_template(template='You are a customer service chatbot. You take in an input support ticket that has a category, priority and a ticket summary attached to it. And return a proper response addressing the pain points of the issue faced by the customer ticket based on the priority. Here is the ticket \n {ticket}')

chain2 = RunnableParallel(
  response=response_prompt | llm | string_parser,
  original_ticket_text=RunnableLambda(lambda x: x['raw_ticket_text'])
)

final_response = chain2.invoke({
  'ticket': ticket,
  'raw_ticket_text': raw_ticket_text
})

print(final_response)