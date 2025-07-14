# Import necessary libraries
import streamlit as st 
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialise Gemini Model
model = genai.GenerativeModel('gemini-2.0-flash')

# Domain prompt that gives context to the LLM
DOMAIN_PROMPT = """
You are a helpful and knowledgeable financial advisor.

Your task is to assist users in optimizing their monthly budgets based on their income, past spending patterns, and specific financial goals.

When the user provides this information, respond with a clear, actionable, and realistic monthly budget plan.

Break the user's income into the following categories:
- Essentials (e.g. rent, groceries, bills)
- Debt Repayment (e.g. loans, EMIs)
- Savings (e.g. emergency fund, short-term needs)
- Investments (e.g. SIPs, mutual funds)
- Lifestyle (e.g. dining, shopping, travel)

Explain how each part supports the user's goals. Make sure your recommendations are practical and aligned with their income level and timelines.

If the user asks a question unrelated to budgeting or financial goals, politely decline.
"""

# Streamlit setup
st.set_page_config(page_title="Goal-Based Budget Optimizer", layout="centered")
st.title("📊 Budget Optimization Assistant")
st.markdown("Tell me your income and goals, or upload your past expenses to get started!")

# CSV Parser
def parse_expense_csv(file_path):
    df = pd.read_csv(file_path)

    # Map CSV column names to expected names
    column_mapping = {
        'Narration / Description': 'Description',
        'Amount (₹)': 'Amount',
        'Balance (₹)': 'Balance',
        'Reference No.': 'Reference Number'
    }

    df = df.rename(columns=column_mapping)

    # Check for required columns
    required_columns = ['Date', 'Description', 'Amount', 'Type', 'Balance', 'Mode', 'Reference Number']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    df['Description'] = df['Description'].astype(str).str.lower()
    df['Type'] = df['Type'].astype(str).str.lower()

    # Filter only debits
    df = df[df['Type'] == 'debit']
    return df

# Store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload section
uploaded_file = st.file_uploader("Upload your expense CSV file", type="csv")
parsed_summary = ""

if uploaded_file is not None:
    try:
        df = parse_expense_csv(uploaded_file)
        total_expense = df['Amount'].sum()
        top_categories = df['Description'].value_counts().head(5).to_string()

        parsed_summary = f"\n\nHere is a summary of your uploaded expenses:\n\n- Total Debits: ₹{total_expense:,.2f}\n- Top expense categories:\n{top_categories}"
        st.success("✅ Expense data uploaded and analyzed.")
        st.markdown(parsed_summary)
    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Hello 👋 What are your goals and monthly income?")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Combine prompt with optional parsed CSV insights
    full_prompt = f"{DOMAIN_PROMPT}\n\nUser Input:\n{user_input}\n{parsed_summary}"

    # Generate Gemini response
    try:
        response = model.generate_content(full_prompt)
        bot_reply = response.text
    except Exception as e:
        bot_reply = f"⚠️ Error: {e}"

    st.chat_message("assistant").markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
