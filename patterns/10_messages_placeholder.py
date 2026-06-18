"""
PATTERN: Chat history with MessagesPlaceholder
DAY: 10

PURPOSE: To store the conversation history in memory
USE WHEN: When you want the LLM to be stateful and remember the conversation history. This allows you to ask follow up questions
FLOW: 
user input
  → load history from memory
  → inject history into prompt
  → LLM
  → save response back to memory
  → output
NEW CONSTRUCTS: MessagesPlacheholder
"""

from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model='gpt-4o-mini')
parser = StrOutputParser()
cpt = ChatPromptTemplate(
    [
        ('system', 'You are a helpful learning assistant for data structures. Keep responses concise'),
        (MessagesPlaceholder(variable_name='chat_history')),
        ('human', '{input}')
    ]
)

chain = cpt | llm | parser
chat_history = []
print(chat_history)


user_input = 'Can you help me understand how to recognize backtracking?'
response = chain.invoke({
    'input': user_input,
    'chat_history': chat_history
})

chat_history.append(HumanMessage(content=user_input))
chat_history.append(AIMessage(content=response))
print(chat_history)
print('-----')


follow_up = 'And what is the usual time and space complexity for this pattern'
response = chain.invoke({
    'input': follow_up,
    'chat_history': chat_history
})

chat_history.append(HumanMessage(follow_up))
chat_history.append(AIMessage(content=response))
print(chat_history)

'''
POST-ANALYSIS BLOCK

The Chat Model expects role tagged messages turn by turn. That's why dumping the chat history as a single variable on human, system or ai prompt
- we lose the message roles - who said what

One-liner
MessagesPlaceholder unpacks the chat history list into a message sequence the model can read with correct roles

'''