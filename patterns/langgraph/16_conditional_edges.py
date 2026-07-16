"""
PATTERN: Conditional Edges
DAY: 16

PURPOSE: Create a conditional graph with LangGraph
USE WHEN: You want to create route control to different nodes based on conditions/rules
FLOW: input → node1 → condition → (node2 or node3 or node4 or END) → final GraphState

NEW CONSTRUCTS: 
add_conditional_edges() - takes in source node and 'routing' function
Routing function has all routing rules. Returns strings/names of the nodes to which control should be routed
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

llm = ChatOpenAI(model='gpt-4o-mini')
string_parser = StrOutputParser()

class SupportTicketClass(BaseModel):
    category: Literal['general', 'technical', 'billing']


class State(TypedDict):
    ticket: str # ticket text submitted by user
    category: Literal['general', 'technical', 'billing'] # issue category
    response: str #


# Node 1
def classify(state: State) -> dict:
    ticket = state['ticket']
    parser = PydanticOutputParser(pydantic_object=SupportTicketClass)

    classify_prompt_template = ChatPromptTemplate(
        [
            ('system', 'You are a customer support assistant that helps classify user submitted support tickets into one of general, technical or billing'),
            ('user', '{ticket}\n {format_instructions}')
        ],
        partial_variables={'format_instructions': parser.get_format_instructions()}
    )

    chain = classify_prompt_template | llm | parser

    response = chain.invoke(
        {'ticket': ticket}
    )

    return {'category': response.category}

def billing_handler(state: State) -> dict:
    ticket = state['ticket']

    billing_prompt_template = ChatPromptTemplate(
        [
            ('system', 'You are a customer support assistant on billing issues. Read the given ticket text and provide appropriate response to the user'),
            ('user', '{ticket}')
        ]
    )
    chain = billing_prompt_template | llm | string_parser
    
    response = chain.invoke({
        'ticket': ticket
    })

    return {'response': response}


def general_handler(state: State) -> dict:
    ticket = state['ticket']

    general_prompt_template = ChatPromptTemplate(
        [
            ('system', 'You are a customer support assistant on general issues. Read the given ticket text and provide appropriate response to the user'),
            ('user', '{ticket}')
        ]
    )
    chain = general_prompt_template | llm | string_parser
    
    response = chain.invoke({
        'ticket': ticket
    })

    return {'response': response}

def technical_handler(state: State) -> dict:
    ticket = state['ticket']

    technical_prompt_template = ChatPromptTemplate(
        [
            ('system', 'You are a customer support assistant on technical issues. Read the given ticket text and provide appropriate response to the user'),
            ('user', '{ticket}')
        ]
    )
    chain = technical_prompt_template | llm | string_parser
    
    response = chain.invoke({
        'ticket': ticket
    })

    return {'response': response}

def route(state: State) -> str:
    category = state['category']

    if category == 'billing':
        return 'billing'
    if category == 'general':
        return 'general'
    if category == 'technical':
        return 'technical'




builder = StateGraph(State)
builder.add_node(classify)
builder.add_node(billing_handler)
builder.add_node(general_handler)
builder.add_node(technical_handler)

builder.add_edge(START, 'classify')
builder.add_conditional_edges('classify', route, {'billing': 'billing_handler', 'general': 'general_handler', 'technical': 'technical_handler'})

builder.add_edge('billing_handler', END)
builder.add_edge('general_handler', END)
builder.add_edge('technical_handler', END)

graph = builder.compile()

response = graph.invoke({'ticket': 'I have been charged twice for a recent purchase. The invoice ids are: inv809210 and inv93021. please arrange a refund!!' })

print(response)


'''
POST-ANALYSIS
State initialization starts during invoke()

Functions read the state and send partial updates to state. But we don't have to manually pass a state instance to any of the node, that's what 
LangGraph is for. Even with the first node that is called on invoke() (the nodes directly tied to START), the state is passed by LangGraph.
We just need to ensure the dict we send on invoke has all the keys that will be needed in the inital nodes. Then if the graph is 
architected correctly (all keys are correctly updated with values before a succeeding node uses it) everything happens bts.

At least when you come to this topic after out of touch, the only touching point where I do womething with the graph is
response = graph.invoke({'ticket': 'jadskkkjasd'})

the dict inside invoke() is what the initial state values would be. And response is the final graph state after execution (reaching END).
You decide what you want to do with response/final state

One liner: Conditional edges route control to one/more nodes from based on specific conditions / routing rules
'''





