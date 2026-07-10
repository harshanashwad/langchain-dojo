'''
BUILD DAY 2
═══════════════════════════════════════════════

Build a chatbot with memory and RAG to answer questions grounded in photosynthesis document

Two chains needed:
One chain to classify query type

Second chain using Runnable Branch to use retriever accordingly based on query type

'''


from dotenv import load_dotenv


load_dotenv()

from operator import itemgetter
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough

# Chain I: Classify query type: general or photosynthesis

chat_history = []

class QueryType(BaseModel):
    query: str = Field(description='The raw query string')
    query_type: Literal['photosynthesis', 'general'] = Field(description='Query type. Can be either photosynthesis or general')

parser = PydanticOutputParser(pydantic_object=QueryType)
llm = ChatOpenAI(model='gpt-4o-mini')

cpt = ChatPromptTemplate(
    messages=[
        ('system', 'You are a query classifier. You read the input string and classify the raw query as either general or photosynthesis. If the conversation history suggests the query is about photosynthesis, classify it as photosynthesis even if the query alone is ambiguous. \n\n {format_instructions}'),
        (MessagesPlaceholder(variable_name='chat_history')),
        ('user', '{query}')
    ], 
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

query_classifier = cpt | llm | parser


# Chain II: RunnableBranch to use retiriever based on query type

# Build RAG Components

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma


text_loader = TextLoader('photosynthesis_research.txt')
documents = text_loader.load() # Returns a list with single document

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=120, separators=['\n\n', '\n', '.', ' '])
chunks = splitter.split_documents(documents=documents)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
chroma_vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = chroma_vector_store.as_retriever()

def format_docs(chunks):
    chunk_texts = [chunk.page_content for chunk in chunks]
    context = '\n\n'.join(chunk_texts)
    return context


string_parser = StrOutputParser()

context_cpt = ChatPromptTemplate(
    [
        ('system', 'Answer the user query based on the context provided \n\n {context}'),
        (MessagesPlaceholder(variable_name='chat_history')),
        ('user', '{query}')
    ]
)

general_cpt = ChatPromptTemplate([
    ('system', 'You are a helpful assistant'),
    (MessagesPlaceholder(variable_name='chat_history')),
    ('user', '{query}')
])

chain = RunnableBranch(
    (lambda x: x['query_type'] == 'photosynthesis', RunnablePassthrough.assign(context=itemgetter('query') | retriever | RunnableLambda(format_docs) ) | context_cpt),
    general_cpt
) | llm | string_parser



# Driver function that invokes the chains with the query and stores chat history

def chatbot_driver(query: str):
    query_classifier_response = query_classifier.invoke(
        {
            'query': query,
            'chat_history': chat_history
        }
    )

    print('Query Classifier Response: ' + query_classifier_response.query_type)

    final_response = chain.invoke(
        {**query_classifier_response.model_dump(), 'chat_history': chat_history}
        )

    # Populate chat_history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=final_response))

    print('Assistant: ' + final_response)


while True:
    query = input('You: ')
    if query.lower() == 'quit':
        break
    print('-'*100)
    chatbot_driver(query)

'''
POST ANALYSIS:

Built a RAG Chatbot with memory. Made my hands dirty with everything needed for 2 important features. 1 is maintaining chat history -> key construct is messages placeholder.
Other is RAG -> built all components needed to load the document chunk it and build a chroma vector store and convert into a retriever instance to use it in a chain.
On top of that built a query classifier and based on that ran a Branch chain to run the query normally or with the retriever path. Both paths retain chat_history!

Two hardest architectural decisions:
1. Split chain approach to query classify query type in one chain and based on that run a branch chain. While it made things a bit complicated with chat history, the architecture was simple.
2. Include chat history in all paths and in both chains so bot is 'aware' of what is going on
'''