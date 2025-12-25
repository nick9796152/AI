# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# GOAL: ADD CHAT INTERFACE FOR CONVERSATIONAL INTERFACE

# streamlit run 05_marketing_assistant_streamlit_02.py

# APP #2: Add Chat Interface
# - Users can now have multiple questions (not limited to single submission)
# - PROBLEM: RAG and memory are not connected


# Key Modifications:
#  1. Persistent Chat History: Utilizing Streamlit's session state, this setup remembers past messages across reruns of the app. Each user interaction and the corresponding assistant response are appended to a message list.
#  2. Using Chat Components for Display: Each message, whether from the user or the AI assistant, is displayed within a st.chat_message context, clearly distinguishing between the participants.



prompt_1 = """
Draft an email to a prospective client to introduce your social media marketing services. Give 3 tips based on experts in the space from our AI database. Transition to the next problem which is how to convert leads into customers. See if the client would like to schedule a 15-minute call to discuss further. 
"""

prompt_2 = """
Modify the prospect's name to Dave. Modify my name is Matt and I'm the Founder of Business Science. 
"""

# Libraries
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import yaml
import streamlit as st

# Initialize the Streamlit app
st.set_page_config(page_title="Your Marketing AI Assistant", layout="wide")
st.title("Your Marketing AI Assistant")

# Load the API Key securely
OPENAI_API_KEY = yaml.safe_load(open('../credentials.yml'))['openai']

# Function to create the processing chain
def create_chain(api_key):
    embedding_function = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        api_key=api_key
    )

    vectorstore = Chroma(
        persist_directory="data/chroma.db",
        embedding_function=embedding_function
    )

    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
{context}

Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(
        model='gpt-3.5-turbo',
        temperature=0.7,
        api_key=api_key
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain



# * NEW: Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "ai", "content": "How can I help you?"}]

# * NEW: Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# * NEW: Chat messages get appended as Q&A occurs in the app
if question := st.chat_input("Enter your Marketing question here:"):
    with st.spinner("Thinking..."): 
        
        # Add user message to chat history
        st.chat_message("human").write(question)

        # Get the response from the AI model
        response = create_chain(OPENAI_API_KEY).invoke(question)
        
        # For Debugging
        print(response)
        print("/n")

        # Add AI response to chat history
        st.chat_message("ai").write(response)

