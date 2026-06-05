"""
PATTERN: RunnablePassthrough
DAY: 5

PURPOSE: Passes input through unchanged to the next step. Used to preserve 
the original input alongside transformed versions of it in a parallel branch.

USE WHEN: When you want to apply a transformation in a step, but also want the original input from previous step. Like in a RAG pipeline,
use the query to retrieve chunks but then, we augment the prompt in next step not just using the chunks but also the original query

FLOW:
{"question": "..."} 
  → RunnableParallel(
      context: RunnableLambda(retrieve_docs),  ← transforms input
      question: RunnablePassthrough()           ← keeps original
    )
  → {"context": "...", "question": "..."}      ← both available for next step
  → prompt | llm | StrOutputParser

NEW CONSTRUCTS: 
RunnablePassthrough()
"""

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')
prompt_template = PromptTemplate.from_template(template='Answer my question elegantly {question} based on the provided context \n {context}')
parser = StrOutputParser()

context="Python variable names must start with a letter (A–Z, a–z) or an underscore (_), followed by any combination of letters, digits (0–9), and underscores. They are case-sensitive, so `score`, `Score`, and `SCORE` are three distinct variables. Reserved keywords like `if`, `for`, `class`, and `return` cannot be used as variable names. Single characters like `x`, `A`, or `_` are valid, and so are longer descriptive names like `total_price` or `userID`. By convention, Python developers use `snake_case` for most variables and `UPPER_SNAKE_CASE` for constants, though these are style guidelines rather than language rules. Invalid names include those starting with a digit (like `2fast`) or containing special characters like hyphens or spaces (like `my-var` or `my var`)."


A = RunnableParallel(
    context=RunnableLambda(lambda x: context),
    question=RunnablePassthrough()
)


chain = A | prompt_template | llm | parser

# response = A.invoke('Is MyName a valid variable name and also follows PEP?')

response = chain.invoke('Is HarshanAshwad a valid variable name and also follows PEP?')

print(response)

'''
POST-ANALYSIS BLOCK
RunnableParallel expects a Runnable. Cannot directly hardcode a string in it. So you wrap a lambda function that returns the string with RunnableLambda and assign your context that.
And each keyword arguments could be processed in parallel and it returns a dict which can be passed on to the chain, like a prompt template (say)

TypeError: Expected a Runnable, callable or dict.Instead got an unsupported type: <class 'str'>

RunnablePassthrough(): A wrapper to just take the input as is and use it within a RunnableParallel block
'''