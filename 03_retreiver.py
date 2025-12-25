# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# Goals: Intro to ... 
# - Document Retrieval
# - Augmenting LLMs with the Expert Information

# LIBRARIES 

from langchain_community.vectorstores import Chroma

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import pandas as pd
import yaml
from pprint import pprint


# OPENAI_API_KEY

OPENAI_API_KEY = yaml.safe_load(open('../credentials.yml'))['openai']


# 1.0 CREATE A RETRIEVER FROM THE VECTORSTORE 

embedding_function = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    api_key=OPENAI_API_KEY
)

vectorstore = Chroma(
    persist_directory="data/chroma.db",
    embedding_function=embedding_function
)

retriever = vectorstore.as_retriever()

retriever

# 2.0 USE THE RETRIEVER TO AUGMENT AN LLM

# * Prompt template 

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# * LLM Specification

model = ChatOpenAI(
    model = 'gpt-3.5-turbo',
    temperature = 0.7,
    api_key=OPENAI_API_KEY
)

# * Combine with Lang Chain Expression Language (LCEL)
#   - Context: Give it access to the retriever
#   - Question: Provide the user question as a pass through from the invoke method
#   - Use LCEL to add a prompt template, model spec, and output parsing

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# * Try it out:

result = rag_chain.invoke("What are the top 3 things needed in a good social media marketing strategy?")

pprint(result)

