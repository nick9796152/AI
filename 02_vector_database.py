# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# Goals: Intro to ... 
# - Langchain Document Loaders
# - Text Embeddings
# - Vector Databases

# LIBRARIES 

from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

import pandas as pd
import yaml
from pprint import pprint

# OPENAI API SETUP

OPENAI_API_KEY = yaml.safe_load(open('../credentials.yml'))['openai']


# 1.0 DATA PREPARATION ----

youtube_df = pd.read_csv('data/youtube_videos.csv')

youtube_df.head()

# * Text Preprocessing

youtube_df['page_content'] = youtube_df['page_content'].str.replace('\n\n', '\n', regex=False)

# * Document Loaders
#   https://python.langchain.com/docs/integrations/document_loaders/pandas_dataframe 

loader = DataFrameLoader(youtube_df, page_content_column='page_content')

documents = loader.load()

documents[0].metadata
documents[0].page_content

pprint(documents[0].page_content)

len(documents)

# * Text Splitting
#   https://python.langchain.com/docs/modules/data_connection/document_transformers

CHUNK_SIZE = 1000

# Character Splitter: Splits on simple default of 
text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    # chunk_overlap=100,
    separator="\n"
)

docs = text_splitter.split_documents(documents)

docs[0].metadata

len(docs)

# Recursive Character Splitter: Uses "smart" splitting, and recursively tries to split until text is small enough
text_splitter_recursive = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE,
    chunk_overlap=100,
)

docs_recursive = text_splitter_recursive.split_documents(documents)

len(docs_recursive)

# * Text Embeddings

# OpenAI Embeddings
# - See Account Limits for models: https://platform.openai.com/account/limits
# - See billing to add to your credit balance: https://platform.openai.com/account/billing/overview

embedding_function = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    api_key=OPENAI_API_KEY
)

# Open Source Alternative:
# Requires Torch and SentenceTransformer packages:
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# * Langchain Vector Store: Chroma DB
# https://python.langchain.com/docs/integrations/vectorstores/chroma

# Creates a sqlite database called vector_store.db
vectorstore = Chroma.from_documents(
    docs, 
    embedding=embedding_function, 
    persist_directory="data/chroma_2.db"
)

vectorstore


# * Similarity Search: The whole reason we did this

result = vectorstore.similarity_search("How to create a social media strategy", k = 4)

pprint(result[0].page_content)


