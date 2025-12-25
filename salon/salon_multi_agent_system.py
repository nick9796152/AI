# SALON MULTI-AGENT SYSTEM
# STEP 3: COMPLETE MULTI-AGENT SYSTEM
# ***

# Goal: Create a comprehensive AI copilot for salon business management
# Combines: Service Expert (RAG), BI Expert (SQL), Customer Scoring (ML), Marketing Writer

# =============================================================================
# LIBRARIES
# =============================================================================

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from langgraph.graph import StateGraph, END

# Use Claude (Anthropic) instead of OpenAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

from typing import Annotated, Sequence, TypedDict

import operator
import pandas as pd
import numpy as np
import sqlalchemy as sql
import plotly.express as px
import plotly.io as pio

import os
import yaml
import ast
import json
import re

from pprint import pprint

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
PATH_SERVICES_VECTORDB = "data/salon_services_vectordb"
PATH_TRANSACTIONS_DATABASE = "sqlite:///data/salon.db"

# API Setup - Load Anthropic API Key
credentials = yaml.safe_load(open('../credentials.yml'))
os.environ["ANTHROPIC_API_KEY"] = credentials['anthropic']

# LLM - Using Claude
CLAUDE_LLM = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=4096)

# =============================================================================
# SUPERVISOR AGENT
# =============================================================================

subagent_names = [
    "Service_Expert",
    "Business_Intelligence_Expert",
    "Customer_Scoring_Expert",
    "Marketing_Email_Writer"
]

def create_supervisor_agent(subagent_names: list, llm):
    """Create the supervisor that routes between agents - Claude compatible"""

    route_options = ["FINISH"] + subagent_names

    system_prompt = f"""You are a supervisor managing a conversation for a SALON BUSINESS between these workers: {", ".join(subagent_names)}.

Each worker has specific knowledge and skills:

1. Service_Expert: Knows all salon services, prices, durations, and can explain what each service includes. Can answer questions about haircuts, color, treatments, styling, wax, and extensions. Does NOT have access to customer data.

2. Business_Intelligence_Expert: Has access to the salon's transaction database. Can write SQL queries to analyze:
   - Revenue by service, category, time period
   - Customer visit patterns and spending
   - Popular services and trends
   - Customer summaries and segments
   Can produce tables and charts.

3. Customer_Scoring_Expert: Can analyze customer data to:
   - Identify customers at risk of churning (haven't visited recently)
   - Find upsell opportunities (customers who might try new services)
   - Calculate customer lifetime value
   - Segment customers by behavior

4. Marketing_Email_Writer: Writes marketing emails for the salon including:
   - Re-engagement campaigns for inactive customers
   - Promotional emails for services
   - Thank you and loyalty emails
   - Seasonal promotions
   Uses insights from other experts to personalize messages.

Given the user request, respond with ONLY the name of the worker to act next.
Each worker will perform a task and respond with results.
When the task is complete, respond with FINISH.

You must respond with exactly one of these options: {route_options}
Respond with ONLY the option name, nothing else."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Based on the conversation, who should act next? Respond with only the worker name or FINISH."),
    ])

    def parse_response(response):
        """Parse the LLM response to extract the routing decision"""
        content = response.content.strip()
        # Check for exact matches
        for option in route_options:
            if option.lower() in content.lower():
                return {"next": option}
        # Default to FINISH if unclear
        return {"next": "FINISH"}

    supervisor_chain = prompt | llm | parse_response

    return supervisor_chain


supervisor_agent = create_supervisor_agent(subagent_names=subagent_names, llm=CLAUDE_LLM)

# =============================================================================
# SERVICE EXPERT AGENT (RAG)
# =============================================================================

def create_service_expert_agent(db_path, llm, temperature=0):
    """Create the Service Expert RAG agent"""

    # Use HuggingFace embeddings (free, no API key needed)
    embedding_function = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    llm.temperature = temperature

    vectorstore = Chroma(
        embedding_function=embedding_function,
        persist_directory=db_path
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Context-aware retriever for follow-up questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given a chat history and the latest user question about salon services,
        formulate a standalone question that can be understood without the chat history.
        Do NOT answer the question, just reformulate it if needed."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # QA prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a knowledgeable salon service expert. Use the following information about our salon services to answer questions.
        Be helpful, friendly, and informative. If you don't know something, say so.
        Include relevant details like prices, duration, and recommendations when appropriate.

        {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


service_expert_agent = create_service_expert_agent(PATH_SERVICES_VECTORDB, llm=CLAUDE_LLM, temperature=0.7)

# =============================================================================
# BUSINESS INTELLIGENCE EXPERT AGENT
# =============================================================================

def create_business_intelligence_agent(db_path, llm, temperature=0):
    """Create the BI Expert with SQL capabilities and charting"""

    llm.temperature = temperature

    # SQL Database connection
    db = SQLDatabase.from_uri(db_path)

    # SQL Output Parser
    def extract_sql_code(text):
        sql_code_match = re.search(r'```sql(.*?)```', text, re.DOTALL)
        if sql_code_match:
            return sql_code_match.group(1).strip()
        sql_code_match = re.search(r"sql(.*?)'", text, re.DOTALL)
        if sql_code_match:
            return sql_code_match.group(1).strip()
        return text

    class SQLOutputParser(BaseOutputParser):
        def parse(self, text: str):
            sql_code = extract_sql_code(text)
            return sql_code if sql_code else text

    # Routing preprocessor (chart vs table)
    routing_prompt = PromptTemplate(
        template="""
        Determine the output format for this salon analytics question:

        1. Extract the SQL-relevant part of the question (remove chart/table formatting requests)
        2. Decide if user wants a 'chart' or 'table'

        If user mentions: bar chart, line chart, pie chart, donut, visualization, plot, graph -> 'chart'
        If user mentions: list, table, show me, data, numbers, or doesn't specify -> 'table'

        Return JSON with 'formatted_question_sql' and 'output_format'.

        Question: {question}
        """,
        input_variables=["question"]
    )
    routing_preprocessor = routing_prompt | llm | JsonOutputParser()

    # SQL Generator
    sql_prompt = PromptTemplate(
        input_variables=['input', 'table_info', 'top_k'],
        template="""
        You are a SQLite expert for a SALON database. Create a syntactically correct query.

        Available tables and views:
        - transactions: customer_id, transaction_id, date, time, itemization_type, category, item, gross_sales, payment_method, card_brand, device_name
        - customer_summary: customer_id, total_visits, total_services, total_spent, avg_service_price, first_visit, last_visit, days_since_last_visit, unique_categories
        - service_popularity: category, item, times_purchased, unique_customers, total_revenue, avg_price
        - monthly_revenue: month, category, unique_customers, total_services, revenue

        Common salon categories: Haircuts, Color, Treatments, Styling, Wax, Extensions

        Return SQL in ```sql ``` format.
        Do not use LIMIT unless user specifies.
        Use appropriate aggregations for salon analytics.

        Tables: {table_info}
        Question: {input}
        """
    )

    sql_generator = (
        create_sql_query_chain(llm=llm, db=db, k=int(1e7), prompt=sql_prompt)
        | SQLOutputParser()
    )

    # Database connection for query execution
    sql_engine = sql.create_engine(db_path)
    conn = sql_engine.connect()

    # Chart generator
    repl = PythonREPL()

    chart_prompt = PromptTemplate(
        template="""
        Create a Plotly visualization for this salon data.

        Instructions: {instructions}
        Data (as dict): {data}

        Requirements:
        - Convert data dict to pandas DataFrame
        - Use plotly.express or plotly.graph_objects
        - Use a clean, professional color scheme
        - Add clear title and axis labels
        - Output as JSON: fig_json = pio.to_json(fig); fig_dict = json.loads(fig_json); print(fig_dict)

        Include these imports: import pandas as pd, import plotly.express as px, import plotly.io as pio, import json
        """,
        input_variables=["instructions", "data"]
    )

    @tool
    def python_repl_tool(code: Annotated[str, "Python code to execute"]):
        """Execute python code for chart generation"""
        try:
            result = repl.run(code)
        except BaseException as e:
            return f"Error: {repr(e)}"
        return result

    chart_generator = chart_prompt | llm.bind_tools([python_repl_tool])

    # Summarizer
    summarizer_prompt = PromptTemplate(
        template="""
        Summarize this salon analytics result in business-friendly terms.

        Question: {question}
        SQL Query: {sql_query}
        Data: {data}
        Output Type: {output_type}

        Provide insights relevant to running a salon business.
        Keep it concise but highlight key findings.
        """,
        input_variables=["question", "sql_query", "data", "output_type"]
    )
    summarizer = summarizer_prompt | llm | StrOutputParser()

    # Build the BI workflow
    class BIState(TypedDict):
        question: str
        formatted_question: str
        output_format: str
        sql_query: str
        data: dict
        chart_json: str
        summary: str

    def route_question(state):
        result = routing_preprocessor.invoke({"question": state["question"]})
        return {
            "formatted_question": result.get('formatted_question_sql', state["question"]),
            "output_format": result.get('output_format', 'table')
        }

    def generate_sql(state):
        sql_query = sql_generator.invoke({"question": state["formatted_question"] or state["question"]})
        return {"sql_query": sql_query.rstrip("'")}

    def execute_query(state):
        df = pd.read_sql(state["sql_query"], conn)
        return {"data": df.to_dict()}

    def should_chart(state):
        return "chart" if state.get('output_format') == "chart" else "table"

    def generate_chart(state):
        instructions = f"Create a chart for: {state['question']}"
        response = chart_generator.invoke({"instructions": instructions, "data": state["data"]})
        try:
            code = dict(response)['tool_calls'][0]['args']['code']
            result = repl.run(code)
            return {"chart_json": result}
        except:
            return {"chart_json": "Chart generation failed"}

    def summarize(state):
        summary = summarizer.invoke({
            "question": state["question"],
            "sql_query": state["sql_query"],
            "data": pd.DataFrame(state["data"]).head(20).to_string(),
            "output_type": state["output_format"]
        })
        return {"summary": summary}

    # Build graph
    workflow = StateGraph(BIState)
    workflow.add_node("route", route_question)
    workflow.add_node("sql", generate_sql)
    workflow.add_node("execute", execute_query)
    workflow.add_node("chart", generate_chart)
    workflow.add_node("summarize", summarize)

    workflow.set_entry_point("route")
    workflow.add_edge("route", "sql")
    workflow.add_edge("sql", "execute")
    workflow.add_conditional_edges("execute", should_chart, {"chart": "chart", "table": "summarize"})
    workflow.add_edge("chart", "summarize")
    workflow.add_edge("summarize", END)

    return workflow.compile()


business_intelligence_agent = create_business_intelligence_agent(
    db_path=PATH_TRANSACTIONS_DATABASE,
    llm=CLAUDE_LLM,
    temperature=0
)

# =============================================================================
# CUSTOMER SCORING EXPERT AGENT
# =============================================================================

def create_customer_scoring_agent(db_path, llm, temperature=0):
    """Create Customer Scoring agent for churn prediction and segmentation"""

    llm.temperature = temperature
    sql_engine = sql.create_engine(db_path)

    scoring_prompt = PromptTemplate(
        template="""
        You are a Customer Scoring Expert for a salon. Analyze customer data to identify:

        1. CHURN RISK: Customers who haven't visited recently (high days_since_last_visit)
           - Low Risk: < 45 days
           - Medium Risk: 45-90 days
           - High Risk: > 90 days

        2. CUSTOMER VALUE: Based on total_spent and total_visits
           - VIP: Top 10% by spending
           - Regular: Middle 60%
           - Occasional: Bottom 30%

        3. UPSELL OPPORTUNITIES: Customers using few service categories (unique_categories)
           - If unique_categories = 1: High upsell potential
           - If unique_categories = 2-3: Medium upsell potential
           - If unique_categories > 3: Already diversified

        4. SERVICE AFFINITY: What services does each customer prefer?

        Based on the request, query the customer_summary view and transactions table,
        then provide scored/segmented customer lists with actionable insights.

        User Request: {request}
        Chat History: {chat_history}

        Provide a detailed analysis with specific customer IDs and recommendations.
        Format the response clearly with sections for each insight type requested.
        """,
        input_variables=["request", "chat_history"]
    )

    def score_customers(request: str, chat_history: list = None):
        """Execute customer scoring analysis"""

        conn = sql_engine.connect()

        # Get customer summary data
        customers_df = pd.read_sql("SELECT * FROM customer_summary", conn)

        # Calculate scores
        customers_df['churn_risk'] = pd.cut(
            customers_df['days_since_last_visit'],
            bins=[-1, 45, 90, 999],
            labels=['Low', 'Medium', 'High']
        )

        # Value segmentation
        spend_threshold_vip = customers_df['total_spent'].quantile(0.9)
        spend_threshold_regular = customers_df['total_spent'].quantile(0.3)

        customers_df['value_segment'] = customers_df['total_spent'].apply(
            lambda x: 'VIP' if x >= spend_threshold_vip
            else ('Regular' if x >= spend_threshold_regular else 'Occasional')
        )

        # Upsell potential
        customers_df['upsell_potential'] = customers_df['unique_categories'].apply(
            lambda x: 'High' if x == 1 else ('Medium' if x <= 3 else 'Low')
        )

        conn.close()

        # Generate analysis based on request
        analysis = scoring_prompt | llm | StrOutputParser()

        # Add data context to request
        data_summary = f"""
        Customer Data Summary:
        - Total Customers: {len(customers_df)}
        - High Churn Risk: {len(customers_df[customers_df['churn_risk'] == 'High'])}
        - Medium Churn Risk: {len(customers_df[customers_df['churn_risk'] == 'Medium'])}
        - VIP Customers: {len(customers_df[customers_df['value_segment'] == 'VIP'])}
        - High Upsell Potential: {len(customers_df[customers_df['upsell_potential'] == 'High'])}

        Sample High-Risk Customers:
        {customers_df[customers_df['churn_risk'] == 'High'].head(10).to_string()}

        Sample VIP Customers:
        {customers_df[customers_df['value_segment'] == 'VIP'].head(10).to_string()}

        Sample High Upsell Potential:
        {customers_df[customers_df['upsell_potential'] == 'High'].head(10).to_string()}
        """

        result = analysis.invoke({
            "request": request + "\n\n" + data_summary,
            "chat_history": str(chat_history or [])
        })

        return result, customers_df

    return score_customers


customer_scoring_agent = create_customer_scoring_agent(
    db_path=PATH_TRANSACTIONS_DATABASE,
    llm=CLAUDE_LLM,
    temperature=0
)

# =============================================================================
# MARKETING EMAIL WRITER AGENT
# =============================================================================

def create_marketing_email_agent(llm, temperature=1.0):
    """Create Marketing Email Writer agent"""

    llm.temperature = temperature

    marketing_prompt = PromptTemplate(
        template="""
        You are an expert marketing copywriter for a salon. Write compelling, personalized emails.

        SALON BRAND VOICE:
        - Friendly and welcoming
        - Professional but not stuffy
        - Passionate about hair and beauty
        - Focused on making clients feel confident

        Based on the conversation context, write a marketing email that includes:

        1. SUBJECT LINE: Attention-grabbing, under 50 characters
        2. PREVIEW TEXT: Compelling preview for inbox, under 100 characters
        3. EMAIL BODY:
           - Personalized greeting
           - Main message/offer
           - Clear call-to-action
           - Warm closing

        EMAIL TYPES:
        - Re-engagement: "We miss you!" for inactive customers
        - Upsell: Introduce new services to existing customers
        - Loyalty: Thank VIP customers, offer rewards
        - Seasonal: Holiday promotions, summer specials
        - New Service: Announce new offerings

        IMPORTANT:
        - If targeting specific customers, mention the segment (e.g., "For our color clients...")
        - Include a clear offer when appropriate (%, $off, free add-on)
        - Always include booking CTA

        User Request: {request}
        Context from other experts: {context}
        """,
        input_variables=["request", "context"]
    )

    marketing_agent = marketing_prompt | llm | StrOutputParser()

    return marketing_agent


marketing_email_agent = create_marketing_email_agent(llm=CLAUDE_LLM, temperature=1.0)

# =============================================================================
# LANGGRAPH MAIN WORKFLOW
# =============================================================================

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    num_steps: Annotated[Sequence[int], operator.add]
    next: str
    customer_data: dict  # Store customer scoring data for marketing use


def get_last_human_message(msgs):
    for msg in reversed(msgs):
        if isinstance(msg, HumanMessage):
            return msg
    return None


def get_last_ai_message(msgs, target_name=None):
    for msg in reversed(msgs):
        if isinstance(msg, AIMessage):
            if not target_name or msg.name == target_name:
                return msg
    return None


# Node functions
def supervisor_node(state):
    print("---SUPERVISOR---")
    result = supervisor_agent.invoke(state)
    print(f"Routing to: {result['next']}")
    return {'next': result['next'], 'num_steps': [1]}


def service_expert_node(state):
    print("---SERVICE EXPERT---")
    messages = state.get("messages", [])
    last_question = get_last_human_message(messages)

    if last_question:
        result = service_expert_agent.invoke({
            "input": last_question.content,
            "chat_history": messages
        })
        return {
            "messages": [AIMessage(content=result['answer'], name='Service_Expert')],
            'num_steps': [1]
        }
    return {"messages": [], 'num_steps': [1]}


def business_intelligence_node(state):
    print("---BUSINESS INTELLIGENCE EXPERT---")
    messages = state.get("messages", [])
    last_question = get_last_human_message(messages)

    if last_question:
        result = business_intelligence_agent.invoke({"question": last_question.content})

        response_content = result.get('summary', 'Analysis complete.')

        # Include data preview if available
        if result.get('data'):
            df = pd.DataFrame(result['data'])
            if len(df) <= 20:
                response_content += f"\n\nData:\n{df.to_string()}"
            else:
                response_content += f"\n\nData Preview (first 20 rows):\n{df.head(20).to_string()}"

        return {
            "messages": [AIMessage(
                content=response_content,
                additional_kwargs=result,
                name='Business_Intelligence_Expert'
            )],
            'num_steps': [1]
        }
    return {"messages": [], 'num_steps': [1]}


def customer_scoring_node(state):
    print("---CUSTOMER SCORING EXPERT---")
    messages = state.get("messages", [])
    last_question = get_last_human_message(messages)

    if last_question:
        result, customer_df = customer_scoring_agent(last_question.content, messages)

        return {
            "messages": [AIMessage(content=result, name='Customer_Scoring_Expert')],
            "customer_data": customer_df.to_dict(),
            'num_steps': [1]
        }
    return {"messages": [], 'num_steps': [1]}


def marketing_email_node(state):
    print("---MARKETING EMAIL WRITER---")
    messages = state.get("messages", [])
    last_question = get_last_human_message(messages)

    # Gather context from previous agent responses
    context_parts = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.name in ['Service_Expert', 'Business_Intelligence_Expert', 'Customer_Scoring_Expert']:
            context_parts.append(f"{msg.name}: {msg.content[:500]}")

    context = "\n\n".join(context_parts) if context_parts else "No additional context available."

    if last_question:
        result = marketing_email_agent.invoke({
            "request": last_question.content,
            "context": context
        })

        return {
            "messages": [AIMessage(content=result, name='Marketing_Email_Writer')],
            'num_steps': [1]
        }
    return {"messages": [], 'num_steps': [1]}


# Build the workflow
workflow = StateGraph(GraphState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("Service_Expert", service_expert_node)
workflow.add_node("Business_Intelligence_Expert", business_intelligence_node)
workflow.add_node("Customer_Scoring_Expert", customer_scoring_node)
workflow.add_node("Marketing_Email_Writer", marketing_email_node)

# All agents report back to supervisor
for agent in subagent_names:
    workflow.add_edge(agent, "supervisor")

# Supervisor routes to agents or ends
conditional_map = {name: name for name in subagent_names}
conditional_map['FINISH'] = END

workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")

# Compile the app
salon_copilot = workflow.compile()

# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":

    print("\n" + "="*60)
    print("SALON MULTI-AGENT SYSTEM - TESTING")
    print("="*60)

    # Test 1: Service Question
    print("\n--- TEST 1: Service Expert ---")
    result = salon_copilot.invoke(
        {"messages": [HumanMessage(content="What color services do you offer and what's the price range?")]},
        config={"recursion_limit": 10}
    )
    print(get_last_ai_message(result['messages']).content)

    # Test 2: Business Intelligence
    print("\n--- TEST 2: BI Expert ---")
    result = salon_copilot.invoke(
        {"messages": [HumanMessage(content="What are our top 5 services by revenue?")]},
        config={"recursion_limit": 10}
    )
    print(get_last_ai_message(result['messages'], 'Business_Intelligence_Expert').content)

    # Test 3: Customer Scoring
    print("\n--- TEST 3: Customer Scoring ---")
    result = salon_copilot.invoke(
        {"messages": [HumanMessage(content="Find customers at high risk of churning")]},
        config={"recursion_limit": 10}
    )
    print(get_last_ai_message(result['messages'], 'Customer_Scoring_Expert').content)

    # Test 4: Multi-Agent Workflow
    print("\n--- TEST 4: Full Workflow ---")
    result = salon_copilot.invoke(
        {"messages": [HumanMessage(content="""
            Find customers who haven't visited in over 60 days and usually get haircut services.
            Write a re-engagement email offering them 20% off their next color service to encourage them to try something new.
        """)]},
        config={"recursion_limit": 15}
    )

    print("\n--- FULL CONVERSATION ---")
    for msg in result['messages']:
        if isinstance(msg, AIMessage):
            print(f"\n[{msg.name}]:")
            print(msg.content[:1000])
            print("...")
