# AI-Powered-Business-Intelligence-Dashboard
An interactive, multi-functional data analytics dashboard built with Streamlit, leveraging LLMs, machine learning, SQL, Power BI-style visualizations, and automated data profiling. This project enables users to upload datasets (CSV, Excel, PDF, DOCX) and instantly generate insights, visualizations, and AI-powered Q&amp;A—all in one place.


🔍 Features
File Upload: Upload CSV, XLSX, PDF, or DOCX files.

Automated Data Profiling: Generate a comprehensive data profile report using ydata-profiling.

Smart Data Storage: Automatically store uploaded tabular data in an SQLite database with unique table names.

Dynamic Visualizations: Interactive charts powered by Plotly and Streamlit.

LLM-Driven Analysis: Ask natural language questions about your dataset and get answers from a LLaMA-based LLM using Groq API.

Extensible Architecture: Designed to easily incorporate ML models, time series forecasting, anomaly detection, and more.

🛠 Tech Stack
Python, Pandas, SQLAlchemy, Streamlit

LLMs (Meta LLaMA 4 via Groq API)

YData Profiling, Plotly Express

SQLite (via SQLAlchemy ORM)

📈 Ideal Use Cases
Exploratory Data Analysis (EDA)

Business Intelligence & Reporting

AI-augmented data querying

Interactive client dashboards

Automated data understanding for non-technical users

🚀 Coming Soon
Support for uploading multiple files with session tracking

Forecasting & anomaly detection modules

User authentication for file history & persistence

Power BI export integration

