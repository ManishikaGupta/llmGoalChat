# Import necessary libraries
import streamlit as st 
import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Initialise Gemini Model
model=genai.GenerativeModel('gemini-2.0-flash')

# Domain prompt that gives context to the LLM
DOMAIN_PROMPT = """
You are a helpful and knowledgeable financial advisor.

Your task is to assist users in optimizing their monthly budgets based on their income and specific financial goals. When the user provides this information, respond with a clear, actionable, and realistic monthly budget plan.

Break the user's income into the following categories:
- Essentials (e.g. rent, groceries, bills)
- Debt Repayment (e.g. loans, EMIs)
- Savings (e.g. emergency fund, short-term needs)
- Investments (e.g. SIPs, mutual funds)
- Lifestyle (e.g. dining, shopping, travel)

Explain how each part supports the user's goals. Make sure your recommendations are practical and aligned with their income level and timelines.
If the user asks a question that is unrelated to budgeting, income planning, or financial goal management, **politely decline to answer** and remind them that you are designed specifically to help with financial budgeting and goal planning.
"""

# Streamlit setup
st.set_page_config(page_title="Goal-Based Budget Optimizer", layout="centered")
st.title("📊 Budget Optimization Assistant")
st.markdown("Tell me your income and goals, and I'll help you build a smart budget!")

# Store conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Hello 👋 What are your goals and monthly income?")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Combine domain prompt with user input
    full_prompt = f"{DOMAIN_PROMPT}\n\nUser: {user_input}"

    # Generate Gemini response
    try:
        response = model.generate_content(full_prompt)
        bot_reply = response.text
    except Exception as e:
        bot_reply = f"⚠️ Error: {e}"

    # Show assistant response
    st.chat_message("assistant").markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
