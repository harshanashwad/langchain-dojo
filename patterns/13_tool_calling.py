"""
PATTERN: Tool Definition + Tool Calling
DAY: 13

PURPOSE: Tools are functions the LLM can call to accomplish certain tasks if it fits the use case
USE WHEN: If certain functions could be called and the output could be used by the LLM to give a high quality response, expose tools!

FLOW: user message
-> LLM sees the message + available tools
-> LLM decides if it needs to use one or more tools
-> Call tool X with few args
-> Python code executes the tool, the output is made visible to the LLM
-> LLM returns a response

NEW CONSTRUCTS:
@tool decorator: Make any function Langchain aware
bind_tools([list of tools]) -> this is called on the llm object
"""

from dotenv import load_dotenv
load_dotenv()


from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage


# Define the tools

@tool
def get_word_count(text: str) -> int:
    '''
    Returns the number of words in the given string
    '''
    return len(text.split())

@tool
def get_character_count(text: str) -> int:
    '''
    Returns the number of characters in the given string
    '''
    return len(text)

# print(get_character_count.name, get_character_count.description)
# print(get_character_count.args_schema.schema())

llm = ChatOpenAI(model='gpt-4o-mini')

llm_with_tools = llm.bind_tools([get_character_count, get_word_count])
response = llm_with_tools.invoke("How many words are in this sentence: 'The quick brown fox jumps over the lazy dog'")


# Execute the tool
tool_call = response.tool_calls[0]
# invoking with this dict: {'text': 'The quick brown fox jumps over the lazy dog'}
tool_result = get_word_count.invoke(tool_call['args'])

print(tool_result)

# Feed result back to LLM
messages = [
    HumanMessage("How many words are in: 'The quick brown fox jumps over the lazy dog'"),
    response,  # the AIMessage with tool_calls
    ToolMessage(content=str(tool_result), tool_call_id=tool_call['id'])
]

final_response = llm_with_tools.invoke(messages)
print(final_response.content)

'''
POST-ANALYSIS BLOCK

@tool decorator makes any function a tool that is LangChain aware. LLM can use tools using bind_tools([list_of_tools])
name and description are two keys to get the function name and description (docstring)

Invoke llm with tool access with a query. If llm decided to use any tool, there is tool_calls key in the response_metadata. We can call that 
function and get the response as a ToolMessage. For final response, we create a list of messages with the tool response and use it to invoke the llm

For static conversation structure starting with a system prompt -> Use ChatPromptTemplate
When tools come in, we don't know beforehand what will be used. So create a list of messages manually and invoke instead of chaining the llm
with a prompt template

One liner:
@tool allows you to define tools and expose them to llm which can decide to use them if needed

'''