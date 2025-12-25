# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# ML + AI BUSINESS INTELLIGENCE (FLOW CONTROL)
# ***

# Goal: Add Pandas Data Frame Conversion Step
# - Extend our DAG Workflow

# Requirements:
# pip install langgraph==0.0.48 langchain_groq==0.5.0


# LIBRARIES

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from langchain.prompts import PromptTemplate

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

from langgraph.graph import END, StateGraph

import os
import yaml
from pprint import pprint
from typing import List, TypedDict

import pandas as pd
import sqlalchemy as sql

from IPython.display import Image

# AI SETUP

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']
os.environ["GROQ_API_KEY"] = yaml.safe_load(open("../credentials.yml"))['groq']

OPENAI_LLM = ChatOpenAI(
    model = "gpt-3.5-turbo"
)

GROQ_LLM = ChatGroq(
    model="llama3-70b-8192",
)

llm = OPENAI_LLM

# * AGENTS

# * SQL Agent

PATH_DB = "sqlite:///database/leads_scored.db"

db = SQLDatabase.from_uri(PATH_DB)

sql_generator = create_sql_query_chain(
    llm = llm,
    db = db,
    k = int(1e7)
)

sql_generator

# * Dataframe Conversion

# New: Pandas sql conversion to data frame
sql_engine = sql.create_engine(PATH_DB)

conn = sql_engine.connect()


# * LANGGRAPH
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    sql_query : str
    data: dict
    num_steps : int


def generate_sql(state):
    print("---GENERATE SQL---")
    question = state.get("question")
    
    num_steps = state.get("num_steps")
    
    num_steps += 1
    
    # Generate SQL
    sql_query = sql_generator.invoke({"question": question})
    
    # pprint(sql_query)
    
    return {"sql_query": sql_query, "num_steps": num_steps}

# New: Create Data Frame
def convert_dataframe(state):
    print("---CONVERT DATA FRAME---")

    sql_query = state.get("sql_query")
    
    num_steps = state.get("num_steps")
    
    num_steps += 1
    
    df = pd.read_sql(sql_query, conn)
    
    return {"data": dict(df), "num_steps": num_steps}
  
    
def state_printer(state):
    """print the state"""
    print("---STATE PRINTER---")
    print(f"question: {state['question']}")
    print(f"SQL Query: {state['sql_query']}")
    print(f"Data: {pd.DataFrame(state['data'])}")
    print(f"Num Steps: {state['num_steps']}")

# * WORKFLOW DAG

workflow = StateGraph(GraphState)

workflow.add_node("generate_sql", generate_sql)
workflow.add_node("convert_dataframe", convert_dataframe)
workflow.add_node("state_printer", state_printer)

workflow.set_entry_point("generate_sql")
workflow.add_edge("generate_sql", "convert_dataframe")
workflow.add_edge("convert_dataframe", "state_printer")
workflow.add_edge("state_printer", END)

app = workflow.compile()

Image(app.get_graph().draw_mermaid_png())

# * TESTING

QUESTION = """
Which 10 customers have the highest p1 probability of purchase?
"""

inputs = {"question": QUESTION, "num_steps": 0}
for s in app.stream(inputs):
    print(s)

# expect error due to pandas conversion only allowed on 1 table at a time
QUESTION = """
What are the first five rows of each table?
"""

inputs = {"question": QUESTION, "num_steps": 0}
for s in app.stream(inputs):
    print(s)
    
QUESTION = """
What are the names of each table?
"""

inputs = {"question": QUESTION, "num_steps": 0}
for s in app.stream(inputs):
    print(s)
    
    
QUESTION = """
Extract the transactions table. Return all rows.
"""

inputs = {"question": QUESTION, "num_steps": 0}
for s in app.stream(inputs):
    print(s)
    
    
QUESTION = """
What are the total sales by month-year?
"""

inputs = {"question": QUESTION, "num_steps": 0}
for s in app.stream(inputs):
    print(s)