'''
PATTERN: Runnable Lambda
DAY: 4

PURPOSE: Pattern is about chaining custom functions in an LCEL chain. RunnableLambda is a wrapper around custom functions so it can be chained together
in LCEL. Can be used to add tools to a chain that can be used by an LLM to 

USE WHEN: LCEL only supports langchain native components to be chained. When you want custom functions to be invoked in sequence in a chain, 
use this wrapper. Best use case is tools to expose custom functionality to our LLM app like web search, running code, executing a db query etc.
Not everything requires us to invoke a model to get something

FLOW: chain = prompt template | RunnableLambda(custom_function) | llm | StrOutputParser

NEW CONSTRUCTS: 
RunnableLambda wrapper -  Wrap custom functions in an LCEL using this class
'''

from langchain_core.runnables import RunnableLambda