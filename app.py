# app.py

# --- 1. Imports ---
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# --- 2. Setup and Configuration ---

# Load environment variables from .env file
load_dotenv()

# --- 3. LangChain Components Setup ---

# Initialize the Generative AI model from LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", # This is the correct model name
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
    convert_system_message_to_human=True 
)
# Create a clean Prompt Template
# This structure helps the model understand its role and the user's input variables.
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful and knowledgeable financial advisor. 
         Your task is to assist users in optimizing their monthly budgets. You will be given their income, goals, and a summary of their past expenses.
         Create a clear, actionable, and realistic monthly budget plan broken into:
         - Essentials (e.g., rent, groceries, bills)
         - Debt Repayment (e.g., loans, EMIs)
         - Savings (e.g., emergency fund, short-term needs)
         - Investments (e.g., SIPs, mutual funds)
         - Lifestyle (e.g., dining, shopping, travel)
         Explain how each part supports the user's goals.
         If the user asks a question unrelated to budgeting or finance, politely decline."""),
        # The 'chat_history' variable will be populated by the Memory module
        ("human", "My Information:\n{user_input}\n\nMy Recent Spending Summary:\n{expense_summary}"),
    ]
)

# Initialize ConversationBufferMemory in Streamlit's session state
# This ensures that memory persists across app reruns (i.e., user interactions)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Create the LLMChain
# This chain links the language model, the prompt template, and the memory together.
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=st.session_state.memory,
    verbose=True  # Set to True to see the full prompt in the terminal for debugging
)

# --- 4. Helper Function for CSV Parsing ---

def parse_expense_csv(file):
    """Parses the uploaded CSV file and returns a pandas DataFrame."""
    try:
        df = pd.read_csv(file)
        
        # Standardize column names
        column_mapping = {
            'Narration / Description': 'Description', 'Amount (₹)': 'Amount', 'Balance (₹)': 'Balance',
            'Reference No.': 'Reference Number'
        }
        df = df.rename(columns=lambda col: column_mapping.get(col, col))

        # Check for required columns
        required_columns = ['Date', 'Description', 'Amount', 'Type']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV is missing the required column: '{col}'")

        # Clean and process data
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        df['Description'] = df['Description'].astype(str).str.lower()
        df['Type'] = df['Type'].astype(str).str.lower()
        
        # Filter for debits (expenses)
        df = df[df['Type'] == 'debit']
        return df
    except Exception as e:
        st.error(f"Error parsing CSV file: {e}")
        return None

# --- 5. Streamlit User Interface ---

# Set page title and favicon
st.set_page_config(page_title="Goal-Based Budget Optimizer", layout="centered", initial_sidebar_state="auto")

# Main title and description
st.title("📊 Your Personal Budget Optimizer")
st.markdown("Provide your financial details, or upload an expense sheet, and I'll help you create a tailored budget plan!")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
if "parsed_summary" not in st.session_state:
    st.session_state.parsed_summary = "No expense data has been provided yet."

# File uploader for expense CSV
with st.sidebar:
    st.header("Upload Expenses")
    uploaded_file = st.file_uploader("Upload your expense CSV file", type="csv")
    if uploaded_file is not None:
        with st.spinner("Analyzing expenses..."):
            df = parse_expense_csv(uploaded_file)
            if df is not None:
                total_expense = df['Amount'].sum()
                top_categories = df['Description'].value_counts().head(5)
                
                summary_str = f"- Total Debits: ₹{total_expense:,.2f}\n- Top 5 Expense Categories:\n"
                for category, count in top_categories.items():
                    summary_str += f"  - {category.title()}: {count} times\n"
                
                # Store summary in session state to persist it
                st.session_state.parsed_summary = summary_str
                st.success("✅ Expense data analyzed successfully!")
            else:
                st.session_state.parsed_summary = "Failed to process the uploaded file."

# Display previous messages from chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input field at the bottom of the page
user_input = st.chat_input("What is your monthly income and financial goal?")

if user_input:
    # Display user's message in the chat
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate and display assistant's response
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            try:
                # Use the LangChain chain to get the response
                # It automatically includes memory and formats the prompt
                response = conversation_chain.predict(
                    user_input=user_input,
                    expense_summary=st.session_state.parsed_summary
                )
                st.markdown(response)
            except Exception as e:
                response = f"⚠️ An error occurred: {e}"
                st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
