'''
PATTERN: itemgetter + RunnablePassthrough.assign
DAY: 9

PURPOSE (itemgetter): funciton to extract key from dict. It is from python's standard library 'operator'
PURPOSE (RunnablePassthrough.assign): To add keys to the input_dict that invoked the chain. 

USE WHEN: When you want to enrich the input_dict before proceeding through the chain. Itemgetter as a clean alternative to using a lambda function
to read something from a dict.

For example,
lambda x: x["question"]
itemgetter("question")

FLOW: 
input_dict
  → RunnablePassthrough.assign(new_key=runnable/callable)
  → enriched input_dict (original keys + new_key)
  → next step

NEW CONSTRUCTS:
RunnablePassthrough.assign
operator.itemgetter
'''
from dotenv import load_dotenv
from langchain_core.runnables.base import chain
load_dotenv()

from multiprocessing import context
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

llm = ChatOpenAI(model='gpt-4o-mini')
parser = StrOutputParser()

bst_context = "A binary search tree is a data structure where each node has at most two children. Left child is always smaller, right child is always larger than the parent node."

# Part 1 — RunnablePassthrough.assign
# prompt needs {question} and {context}
prompt_1 = ChatPromptTemplate.from_messages([
    ('system', 'Answer the question using the context provided'),
    ('human', 'Question: {question}\nContext: {context}')
])

# BUILD chain_1 here using RunnablePassthrough.assign
# invoke with {"question": "What is a BST?", "level": "beginner"}

chain1 = RunnablePassthrough.assign(context=lambda x: bst_context) | prompt_1 | llm | parser
response = chain1.invoke({
    "question": "What is a BST?", "level": "beginner"
})

print(response)


# Part 2 — itemgetter
# prompt needs only {question}
prompt_2 = ChatPromptTemplate.from_messages([
    ('system', 'Answer the question clearly'),
    ('human', '{question}')
])

# BUILD chain_2 here using itemgetter
# invoke with {"question": "What is a BST?", "level": "beginner"}

chain_2 = itemgetter('question') | prompt_2 | llm | parser
chain_2_response = chain_2.invoke(
    {"question": "What is a BST?", "level": "beginner"}
)

print(chain_2_response)

'''
POST-ANALYSIS BLOCK

itemgetter retrieves key from whichever dict invokes it (whether it's start or in the midlle of a chain)
RunnablePassthrough() makes sense only in RunnableParallel. But .assign could be used inside parallel as well as standalone to enrich input dict
as needed

One-liner:
Use RunnablePassthrough.assign(): when you need to add keys to the incoming dict before passing onto the chain
itemgetter: retrieve specific key from whatever dict invokes it. Cleaner code than using a lambda as a callable
'''

