# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# ML + AI BUSINESS INTELLIGENCE (FLOW CONTROL)
# ***

# Goal: Create a basic SQL agent to interact with the database

# LIBRARIES

import pandas as pd
import sqlalchemy as sql

from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain

from langchain_community.utilities import SQLDatabase

import yaml
from pprint import pprint

OPENAI_API_KEY = yaml.safe_load(open('../credentials.yml'))['openai']

PATH_DB = "sqlite:///database/leads_scored.db"

# SQL DATABASE AI AGENT

# * Connecting Langchain to a database

db = SQLDatabase.from_uri(PATH_DB)

db.dialect

db.get_usable_table_names()

db.run("SELECT * FROM leads_scored LIMIT 10;")

# * Generating SQL with LLMs

model = ChatOpenAI(
    model = 'gpt-3.5-turbo',
    temperature = 0.7,
    api_key=OPENAI_API_KEY
)

chain = create_sql_query_chain(model, db)

chain

response = chain.invoke({'question': "which 5 customers have the highest p1 probability of purchase?"})

pprint(response)

pprint(db.run(response))

# * Working with Pandas

sql_engine = sql.create_engine(PATH_DB)

conn = sql_engine.connect()

pd.read_sql(response, conn)

