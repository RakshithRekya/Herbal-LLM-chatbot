import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.3
    )