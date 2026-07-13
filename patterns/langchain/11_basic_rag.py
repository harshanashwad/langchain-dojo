"""
PATTERN: Basic RAG  (Load → Split → Embed → Retrieve)
DAY: 11

PURPOSE: To have our chatbot provise responses grounded in knowledge bases/dcoument store
USE WHEN: When you want chatbot to provide responses based on knowledge grounded in external memory
- knowledge bases, document stores, file systems etc (especially when it's too large to stuff into context window of the model)

FLOW: 
raw text file
  → [load]
  → TextDocument
  → [split]
  → Chunks of chunk size with some overlap
  → [embed]
  → embeddings
  → [store in vector db]
  → query comes in
  → [embed query]
  → [similarity search]
  → top-k relevant chunks

NEW CONSTRUCTS: 

DocumentLoader: TextLoader (I am using a .txt file)
Chunking: RecursiveCharacterTextSplitter for chunking
Embeddings: embeddings model (like openai-text3-small) think dimesnions were 1536
Store it in a vector store: FAISS or chromadb for simplicity?
Retrieval: Cosine similarity
"""

def print_chunks(chunks):
  for chunk in chunks:
    print(chunk)
    print()
    print()

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# Loading
text_loader = TextLoader('photosynthesis_research.txt')
documents = text_loader.load()
content = documents[0].page_content

# Splitting/Chunking (with overlap)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=['\n\n', '\n', '.', ' '])
chunks = splitter.split_documents(documents)


# Embeddings model -- embedding not used explicitly. Chroma handles it internally.
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
print(len(vectors[0]))

# Chroma vector store
# documents is a list of Document objects
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)

# Run similarity search
# filter argument takes a dict to filter using metadata
relevant_chunks = vector_store.similarity_search('What are photosyntheitc pathways evolved in terrestrial plants?')
print_chunks(chunks=relevant_chunks)

'''
POST-ANALYSIS BLOCK

TextLoader.load() returns a list with a single Document object (whatever file that was passed to instantiate the loader)

split_text() takes a string and splits them into chunks. split_documents takes a list of documents and splits them into chunks. Each chunk is 
only a document object but with a slice of the page_content of the original document. in practice this is much more convenient than splitting one string at a time using split_text

from_documents() takes a list of documents and embeds them using the provided embeddings model and saves it in Chroma vector store. Then,
we can run semantic search against our documents using the embeddings of input query

Vector stores are used for external memory (unlimited in theory). It's Critical to implement a RAG pipeline to augment prompts/context with relevant 
info to fill knowledge gaps


'''