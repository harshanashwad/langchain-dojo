"""
PATTERN: RAG chain
DAY: 12

PURPOSE: (one line — what + why)
RAG is needed when you want to integrate knowledge from external vector stores for your chatbot 

USE WHEN:
When there is factual knowledge gaps for your chatbot and you want response grounded in external knowledge, RAG is the way to achieve it

FLOW: 
query ->
vector_store_retriever (to retrieve chunks) ->
Augment prompt with retrieved context ->
LLM ->
Grounded response


NEW CONSTRUCTS: 
as_retirever: Converts your vector store into a vector store retriever that can be used directly in chains for retrieveing relevant documents
"""

from dotenv import load_dotenv
load_dotenv()

from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# LLM and Prompt Template
llm = ChatOpenAI(model='gpt-4o-mini')
cpt = ChatPromptTemplate(
    [
        ('system', 'You are QA chatbot that answers user questions based on given context'),
        ('user', 'Answer the following question based on provided context: \n\n {context}\n\n {question}')
    ]
)
parser = StrOutputParser()

# Loading
text_loader = TextLoader('photosynthesis_research.txt')
documents = text_loader.load()
content = documents[0].page_content

# Splitting/Chunking (with overlap)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=['\n\n', '\n', '.', ' '])
chunks = splitter.split_documents(documents)

# Embeddings model -- embedding not used explicitly. Chroma handles it internally.
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Chroma vector store
# documents is a list of Document objects
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
vector_store_retriever = vector_store.as_retriever()

# Helper to append Document chunks
def format_docs(docs):
    context = "\n\n".join(doc.page_content for doc in docs)
    print('Retrieved context:\n', context)
    print('\n\n')
    return context


chain = RunnablePassthrough.assign(context= itemgetter('question') | vector_store_retriever | RunnableLambda(format_docs)) | cpt | llm | parser
response = chain.invoke({
    'question': 'Statistical analysis of the experiment?'
})

print(response)

'''
POST-ANALYSIS BLOCK
The RunnablePassthrough.assign first populates context using another chain. Gets the question string, invokes retriever using it and then sends 
retrieved chunks to get back a context string. This context string is added to the dict that invoked RunnablePassthrough. So following,
we can invoke the prompt template to get the prompt with question and context, involke llm and complete the chain

One liner: Build components needed for basic rag. Convert vector store into a retriever to make it usable in a chain and it becomes a RAG chain

'''