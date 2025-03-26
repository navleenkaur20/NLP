#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
import pandas as pd
import os


# In[2]:


# Load environment variables
os.environ["OPENAI_API_KEY"] = "your_api_key_here"


# In[3]:


data= pd.read_csv('/Users/navleenkaur/Downloads/Sample - Superstore (1).csv',encoding = 'latin-1')


# In[4]:


# Create the AI Agent
agent = create_pandas_dataframe_agent(
    ChatOpenAI(model="gpt-4", temperature=0),
    data,
    verbose=True,
    allow_dangerous_code=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)


# In[5]:


def ask_agent(question):
    """Function to ask questions to the agent and return the response"""
    response = agent.run({
        "input": question,
        "agent_scratchpad": f"Human: {question}\nAI: To answer this question, I need to use Python to analyze the dataframe. I'll use the python_repl_ast tool.\n\nAction: python_repl_ast\nAction Input: ",
    })
    return response


# In[ ]:


# Streamlit UI
def main():
    st.title("Superstore Data Analysis Dashboard")
    
    st.sidebar.write("### Ask Your Own Question")
    
    user_question = st.sidebar.text_input("Enter your question:")
    if st.sidebar.button("Submit"):
        answer = ask_agent(user_question)
        st.write(f"### Question: {user_question}")
        st.write(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()

