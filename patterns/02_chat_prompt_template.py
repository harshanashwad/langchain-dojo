
'''
PATTERN: ChatPromptTemplate + System/Human Messages
DAY 2


WHAT THIS PATTERN IS:
ChatPromptTemplate is a list of structured messages. Each message has a designated role (system/human/assistant) so the model can understand
who is the speaker of each message

WHEN YOU'D USE THIS:
To set initial prefix of the conversation with roles (like system, human, assistant). This will take the conversation with the model in a certain
direction with constraints you set

WHAT PROBLEM IT SOLVES:
System prompt clearly defines the role of the chatbot, set output constraints, tone and more. With full role control in each message, we can add few 
shot examples as a conversation between the user and assistant which makes the model to perfectly understand the expected output format for sample inputs. 
Helps improve reliabilty of the model in well-defined tasks

THE FLOW:
Create a ChatPromptTemplate which is a list of dictionaries representing a conversation/ Each dictionary is a structured message with a role attached.
Irrespective of the role, each message that supprts dynamic insertion of keywords. {"topic": "black holes", "level": "beginner"},
The system prompt could use {topic}, the user message could use {level} to explain how technical or simple the model's explanation about the topic would be

CONSTRUCTS INTRODUCED TODAY:
ChatPromptTemplate (with messages and designated roles)
'''

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Two variations to create ChatPrompt Template

# I see messages which is a list of tuples being directly passed as an attribute or via the from_messages method. Let's test it and see the type
llm = ChatOpenAI(model='gpt-4o-mini')
messages=[
    ('system', 'You are a helpful assistant. You explain given input topic at {level} level'),
    ('human', 'Can you the explain {topic}?'),
]

cpt = ChatPromptTemplate.from_messages(
    messages=messages
) # from_messages() returns a ChatPromptTemplate

parser = StrOutputParser()

chain = cpt | llm | parser

chain_output = chain.invoke({
    'level': 'Beginner',
    'topic': 'Calculus'
})

print(chain_output)
'''
chat_prompt = cpt.format_messages(
    name='Harshan',
    chatbot_name='Andy'
) # Now this prompt is usable to invoke an llm

print(type(cpt))
print(cpt)
print(chat_prompt)
'''


# POST-ANALYSIS
# ─────────────────────────────────────────────────────────
# ChatPromptTemplate.from_messages():  The from_template equivalent of PromptTemplate. Messages is a sequence of tuples each being a specific role message template and represents a conversation

# ChatPromptTemplate internals:
# ('system', 'You are {name}') looks like a final message but it's actually
# a SystemMessagePromptTemplate — a PromptTemplate with a role attached.
# The tuple is just shorthand. LangChain converts it under the hood.
# Only after invoking with a dict (filling all {variables}) do the templates
# become actual messages (SystemMessage, HumanMessage etc.)
# So: ChatPromptTemplate = list of MessagePromptTemplates, not list of messages.

# .invoke():                       invoke always takes dictionaries! use it on the chain with input of the ChatPromptTemplate needed
# format_messsages():               When you manually want to get finalized list of actual message(s) from the template, use this. equivalent of format() in a PromptTemplate

'''
ChatPromptTemplate with role-sepcific when you want to use a ChatModel while setting the conversation prefix (System prompt, set of messages with few shot 
examples, expected model outputs, output constraints, tone and more)

'''
