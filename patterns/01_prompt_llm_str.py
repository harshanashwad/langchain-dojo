"""
PATTERN: PromptTemplate + LLM + StrOutputParser
DAY: 1
═══════════════════════════════════════════════

WHAT THIS PATTERN IS:
The most basic LangChain chain. A reusable prompt with placeholders,
piped through an LLM, piped through a parser that strips the response
down to a plain string.

WHEN YOU'D USE THIS:
Any time you want to send a templated prompt to an LLM and get a clean
string back — text generation, summarization, simple Q&A, explanations.

WHAT PROBLEM IT SOLVES:
Without this pattern you'd manually format strings (using f-strings), call 
the LLM, then dig into the response object to extract the text. This chains
those three steps into one reusable, readable pipeline.

THE FLOW:
{"topic": "black holes"}
  → PromptTemplate      (fills the placeholder)
  → ChatOpenAI          (returns an AIMessage object)
  → StrOutputParser     (extracts .content as plain string)
  → "Black holes are..."

CONSTRUCTS INTRODUCED TODAY:
- PromptTemplate        reusable prompt with {variables}
- ChatOpenAI            LLM wrapper for gpt-4o-mini
- StrOutputParser       converts AIMessage → plain string
- pipe operator |       chains components, output of A → input of B
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. PromptTemplate
template = "Explain the topic:{topic} in a paragraph"
prompt_template = PromptTemplate.from_template(template=template) # from_template() returns a PromptTemplate

# 2. LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.4)

# 3. StrOutputParser
parser = StrOutputParser() # Extracts text content from model outputs as a str

# Manual chaining
# prompt = prompt_template.format(topic='Physics')
# message = llm.invoke(prompt)
# parsed_message = parser.invoke(message)

# print('raw_message', message)
# print('-' * 50)
# print('parsed_message', parsed_message)

# LCEL equivalent
print('-' * 50)

chain = prompt_template | llm | parser
chain_output = chain.invoke(input='Physics')
print('Chained output')
print(chain_output)


# POST-ANALYSIS
# ─────────────────────────────────────────────────────────
# PromptTemplate.from_template():  from_template() returns a PromptTemplate
# .invoke():                       calls the chain (type RunnableSequence) sending the input parameter to the first component in the chain
# Why StrOutputParser exists:      To extract the content field from model output (the message part alone as a string)

'''
PromptTemplate + LLM + StrOutputParser is the base unit of every LangChain chain — a reusable prompt, a model call, and a clean string out.
'''