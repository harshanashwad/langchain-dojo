"""
PATTERN: Linear Graph
DAY: 15

PURPOSE: Create a linear graph with LangGraph
FLOW: input → node1 → node2 → node3 -> output

NEW CONSTRUCTS: 
TypedDict as GraphState: to define state of graph as a schema
StateGraph - a graph builder that takes in the defined state as the graph schema
node - a python function that reads current graph state and sends partial updates
edge - used to define routing rules across edges

compile() - The graph made by builder is validated and can be invoked after compile


"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# Recommended to instatiate llm globally
llm = ChatOpenAI(model='gpt-4o-mini')
parser = StrOutputParser()

# Define the schema for the graph state using typed Dict
class State(TypedDict):
    topic: str
    explanation: str
    questions: str
    score: str

# Node 1
def explain_topic(state: State) -> State:
    topic = state['topic']
    
    cpt = ChatPromptTemplate(
        [
            ('system', 'You are a assistant that give a one-paragraph explanation of the given topic'),
            ('human', '{topic}')
        ]
    )

    chain = cpt | llm | parser
    response = chain.invoke({
        'topic': topic
    })

    # Return partial state that will be used by the graph and do the state update
    return {'explanation': response}

# Node 2
def quiz_questions(state: State) -> State:
    explanation = state['explanation']

    cpt = ChatPromptTemplate(
        [
            ('system', 'You are a assistant that takes in a one-paragraph explanation about a topic and generates exactly three quiz questions from it'),
            ('human', '{explanation}')
        ]
    )

    chain = cpt | llm | parser
    response = chain.invoke({
        'explanation': explanation
    })

    # Return partial state that will be used by the graph and do the state update
    return {'questions': response}

# Node 3:
def analyze(state: State) -> State:

    topic = state['topic']
    explanation = state['explanation']
    questions = state['questions']

    cpt = ChatPromptTemplate(
        [
            ('system', 'You are a educational content analyzer. You take in a topic, one-paragraph explanation about it and some related quiz questions as input and need to score 1-10 about how educational the content is'),
            ('human', 'Topic: {topic}\nExplanation: {explanation}\nQuestions: {questions}')
        ]
    )

    chain = cpt | llm | parser

    score = chain.invoke({
        'topic': topic,
        'explanation': explanation,
        'questions': questions
    })

    return {'score': score}

# Building and wiring the graph

builder = StateGraph(State)
builder.add_node(explain_topic)
builder.add_node(quiz_questions)
builder.add_node(analyze)

builder.add_edge(START, 'explain_topic')
builder.add_edge('explain_topic', 'quiz_questions')
builder.add_edge('quiz_questions', 'analyze')
builder.add_edge('analyze', END)

graph = builder.compile()

response = graph.invoke(
    {'topic': 'blackholes'}
)

print(response)

'''
POST-ANALYSIS:
Nodes return partial dicts, not full State objects.
LangGraph merges the partial update into the shared state automatically.
Each node only needs to return the keys it changed.

Graph finishes executing when we reach the terminal (END) node.
a StateGraph instance is a builder or template. It's not validated yet. In order to invoke it, we call compile()
invoke() on a graph calls all nodes connected to the START node.

'''