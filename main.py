# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:36:17 2025
@author: Oreoluwa
"""
api_key = "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"
# Imports
import streamlit as st
import pandas as pd
import pdfplumber
import time
import os
import json
from groq import Groq
from docx import Document
import plotly.express as px
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
from sqlalchemy import create_engine, text

# Streamlit page setup
st.set_page_config(layout='wide')
st.title("AI-POWERED BUSINESS INTELLIGENCE DASHBOARD")

# Database engine
engine = create_engine("sqlite:///data_storage.db")

# ------------------ Utility Functions ------------------

def extract_pdf_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def ydata(df):
    with st.spinner("Generating Report..."):
        profile = ProfileReport(df, title="Data Profile", explorative=True)
        components.html(profile.to_html(), height=1500, width=1500, scrolling=True)

def visualization(df, column):
    if df[column].dtype == 'object':
        st.bar_chart(df[column].value_counts())
    else:
        st.plotly_chart(px.histogram(df, x=column))



def chat_with_file(user_question: str, df: pd.DataFrame):

    # Convert a preview of the data to JSON (limit to 50 rows)
    data = df.head(50).to_dict(orient="records")
    data_json = json.dumps(data, indent=2)

    # Construct the prompt for the LLM
    prompt = f"""
You are an AI data analyst. A user has uploaded a dataset and would like to understand it better.
Here is a preview of the dataset (first 50 rows):

{data_json}

Now answer the user's question, based on this data.

User's Question: {user_question}

Answer:
"""

    # Connect to Groq API
    client = Groq(api_key = api_key)

    # Run chat completion
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_completion_tokens=1024,
        top_p=1,
        stream=True
    )

    # Stream the response in Streamlit
    st.markdown("### 🤖 AI Answer")
    response_text = ""
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        response_text += content

    return response_text





# ------------------ File Upload ------------------

uploaded_file = st.file_uploader("Please Upload Your File Here", type=["csv", "pdf", "docx", "xlsx"])



if uploaded_file is not None:
    extension = uploaded_file.name.split(".")[-1].lower()

    if extension == "csv":
        df = pd.read_csv(uploaded_file)

    elif extension == "xlsx":
        df = pd.read_excel(uploaded_file)

    elif extension == "pdf":
        extracted_text = extract_pdf_text(uploaded_file)
        st.subheader("Extracted PDF Text")
        st.text_area("Contents", extracted_text, height=400)

    elif extension == "docx":
        
    
    
        doc = Document(uploaded_file)
        extracted_text = '\n'.join([para.text for para in doc.paragraphs])
        st.subheader("Extracted Word Document Text")
        st.text_area("Contents", extracted_text, height=400)

    else:
        st.error("Unsupported file type.")
        df = None
        
        
    
        

    # ------------------ Report/Visualization ------------------
    if 'df' in locals() and isinstance(df, pd.DataFrame):
        
        
        
        if "chatbot" not in st.session_state:
            st.session_state.chatbot = False
        
        basename = uploaded_file.name.rsplit(".",1)[0].replace(" ","_").lower()
        timestamp = str(int(time.time()))
        table_name = f"{basename}_{timestamp}"
        
        try:
            df.to_sql(table_name,engine, if_exists='replace',index=False)
            st.success(f"{table_name} created in database")
            
        except Exception as e:
            st.write(f"Unable to commit to database {e}")
            
            
        
        with st.sidebar:
            auto_generate = st.button("Auto-Generate Report")
        
        
            
            
            if st.button('Chat With Your File'):
                    st.session_state.chatbot = True
                
        if st.session_state.chatbot:
            with st.expander("💬 Ask the AI About Your Data"):
                user_question = st.text_input("Enter your question")
                        
                        
            if st.button("ASK AI") and user_question:
                response = chat_with_file(user_question, df)
                st.write(response)
                
                
                
            
        if auto_generate:
            st.session_state.chatbot = False
            ydata(df)    
       

            