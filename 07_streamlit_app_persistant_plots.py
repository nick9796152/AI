# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# ML + AI BUSINESS INTELLIGENCE (FLOW CONTROL)
# ***

# streamlit run path_to_streamlit_app.py

# Goal: Create a streamlit chat app with persistent plots


# Imports

import streamlit as st
import plotly.express as px

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Initialize Streamlit App

def create_plot():
    df = px.data.iris()
    fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')
    return fig

st.set_page_config(page_title="Your Business Intelligence AI Copilot", layout="wide")
st.title("Your Business Intelligence AI Copilot")

st.markdown("""
            I'm a handy business intelligence agent that connects up to the leads_scored.db SQLite database that mimics an ERP System for a company. You can ask me Business Intelligence, Customer Analytics, and Data Visualization Questions. I will report the results. 
            """)

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

# Initialize plot storage in session state
if "plots" not in st.session_state:
    st.session_state.plots = []

# Function to display chat messages including Plotly charts
def display_chat_history():
    for i, msg in enumerate(msgs.messages):
        with st.chat_message(msg.type):
            if "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                st.plotly_chart(st.session_state.plots[plot_index])
            else:
                st.write(msg.content)

# Render current messages from StreamlitChatMessageHistory
display_chat_history()

if question := st.chat_input("Enter your question here:", key="query_input"):
    with st.spinner("Thinking..."):
        st.chat_message("human").write(question)
        msgs.add_user_message(question)

        response_text = "creating a plot..."
        response_plot = create_plot()

        # Store the plot and keep its index
        plot_index = len(st.session_state.plots)
        st.session_state.plots.append(response_plot)

        # Store the response text and plot index in the messages
        msgs.add_ai_message(response_text)
        msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")
        
        st.chat_message("ai").write(response_text)
        st.plotly_chart(response_plot)

with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)

