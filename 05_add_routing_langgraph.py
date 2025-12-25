# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# ML + AI BUSINESS INTELLIGENCE (FLOW CONTROL)
# ***

# Goal: Add Routing
# - Routing Preprocessor Agent: Should a chart be returned?
# - Decision Processor: Which path? decide_chart_or_table()
# - Conditional Edges (DAG): How the path gets selected

# Requirements:
# pip install langgraph==0.0.48 langchain_groq==0.5.0


# LIBRARIES

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from langchain.prompts import PromptTemplate

from langchain_core.output_parsers import JsonOutputParser

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

# NEW: Routing Preprocessor Agent
#  IMPORTANT: Used to format the user's question and provide specifications for how the output is returned

routing_preprocessor_prompt = PromptTemplate(
    template="""
    You are an expert in routing decisions for a SQL database agent, a Charting Visualization Agent, and a Pandas Table Agent. Your job is to:
    
    1. Determine what the correct format for a Users Question should be for use with a SQL translator agent 
    2. Determine whether or not a chart should be generated or a table should be returned based on the users question.
    
    Use the following criteria on how to route the the initial user question:
    
    From the incoming user question, remove any details about the format of the final response as either a Chart or Table and return only the important part of the incoming user question that is relevant for the SQL generator agent. This will be the 'formatted_user_question_sql_only'. If 'None' is found, return the original user question.
    
    Next, determine if the user would like a data visualization ('chart') or a 'table' returned with the results of the SQL query. If unknown, not specified or 'None' is found, then select 'table'.  
    
    Return JSON with 'formatted_user_question_sql_only' and 'routing_preprocessor_decision'.
    
    INITIAL_USER_QUESTION: {initial_question}
    """,
    input_variables=["initial_question"]
)

routing_preprocessor = routing_preprocessor_prompt | llm | JsonOutputParser()

routing_preprocessor


# QUESTION = """
# Which 10 customers have the highest p1 probability of purchase?
# """
# response = routing_preprocessor.invoke({"initial_question": QUESTION})
# response
# pprint(response)

# QUESTION = """
# What are the total sales by month-year? Return a chart of sales by month-year. 
# """
# response = routing_preprocessor.invoke({"initial_question": QUESTION})
# response


# QUESTION = """
# What are the total sales by month-year? Please return a table. 
# """
# response = routing_preprocessor.invoke({"initial_question": QUESTION})
# response

# response['formatted_user_question_sql_only']
# response['routing_preprocessor_decision']


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
    
sql_engine = sql.create_engine(PATH_DB)

conn = sql_engine.connect()


# * LANGGRAPH

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    user_question: str
    formatted_user_question_sql_only: str
    sql_query : str
    data: dict
    routing_preprocessor_decision: str
    num_steps : int

# NEW: DETERMINES THE PATH + FORMATS THE QUESTION
def preprocess_routing(state):
    print("---ROUTER---")
    question = state.get("user_question")
    
    num_steps = state.get("num_steps")
    
    num_steps += 1
    
    # Chart Routing and SQL Prep
    response = routing_preprocessor.invoke({"initial_question": question})
    
    formatted_user_question_sql_only = response['formatted_user_question_sql_only']
    
    routing_preprocessor_decision = response['routing_preprocessor_decision']
    
    
    return {
        "formatted_user_question_sql_only": formatted_user_question_sql_only,
        "routing_preprocessor_decision": routing_preprocessor_decision,
        "num_steps": num_steps
    }
    

def generate_sql(state):
    print("---GENERATE SQL---")
    question = state.get("formatted_user_question_sql_only")
    
    # Handle case when formatted_user_question_sql_only is None:
    if question is None:
        question = state.get("user_question")
    
    num_steps = state.get("num_steps")
    
    num_steps += 1
    
    # Generate SQL
    sql_query = sql_generator.invoke({"question": question})
    
    # pprint(sql_query)
    
    return {"sql_query": sql_query, "num_steps": num_steps}


def convert_dataframe(state):
    print("---CONVERT DATA FRAME---")

    sql_query = state.get("sql_query")
    
    num_steps = state.get("num_steps")
    
    num_steps += 1
    
    df = pd.read_sql(sql_query, conn)
    
    return {"data": dict(df), "num_steps": num_steps}

# NEW: Decision Logic

def decide_chart_or_table(state):
    print("---DECIDE CHART OR TABLE---")
    return "chart" if state.get('routing_preprocessor_decision') == "chart" else "table"

def generate_chart(state):
    print("---GENERATE CHART---")
    
    num_steps = state.get("num_steps")
    
    num_steps += 1
    
    # TODO: Add Charting Logic
    
    return {"num_steps": num_steps}
    
    
def state_printer(state):
    """print the state"""
    print("---STATE PRINTER---")
    print(f"User Question: {state['user_question']}")
    print(f"Formatted Question (SQL): {state['formatted_user_question_sql_only']}")
    print(f"SQL Query: \n{state['sql_query']}\n")
    print(f"Chart or Table: {state['routing_preprocessor_decision']}")
    print(f"Data: \n{pd.DataFrame(state['data'])}\n")
    print(f"Num Steps: {state['num_steps']}")

# * WORKFLOW DAG

workflow = StateGraph(GraphState)

workflow.add_node("preprocess_routing", preprocess_routing)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("convert_dataframe", convert_dataframe)
workflow.add_node("generate_chart", generate_chart)
workflow.add_node("state_printer", state_printer)

workflow.set_entry_point("preprocess_routing")
workflow.add_edge("preprocess_routing", "generate_sql")
workflow.add_edge("generate_sql", "convert_dataframe")

# NEW: Conditional Edges

workflow.add_conditional_edges(
    "convert_dataframe", 
    decide_chart_or_table,
    {
        # Result : Step Name To Go To
        "chart":"generate_chart", # Path Chart
        "table":"state_printer" # Path State Printer
    }
)

# workflow.add_edge("convert_dataframe", "state_printer")
workflow.add_edge("generate_chart", "state_printer")
workflow.add_edge("state_printer", END)

app = workflow.compile()

Image(app.get_graph().draw_mermaid_png())

# * TESTING

QUESTION = """
Which 10 customers have the highest p1 probability of purchase?
"""
inputs = {"user_question": QUESTION, "num_steps": 0}
for s in app.stream(inputs):
    print(s)

# expect error due to pandas conversion only allowed on 1 table at a time
QUESTION = """
What are the first five rows of each table?
"""
inputs = {"user_question": QUESTION, "num_steps": 0}
for s in app.stream(inputs):
    print(s)
    
QUESTION = """
What are the names of each table?
"""
inputs = {"user_question": QUESTION, "num_steps": 0}
for s in app.stream(inputs):
    print(s)
    
    
QUESTION = """
Extract the transactions table. Return all rows.
"""
inputs = {"user_question": QUESTION, "num_steps": 0}
for s in app.stream(inputs):
    print(s)
    
    
QUESTION = """
What are the total sales by month-year? Make a chart of sales over time.
"""
inputs = {"user_question": QUESTION, "num_steps": 0}
for s in app.stream(inputs):
    print(s)
    
