import os
import logging
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

PROVIDERS = [
    {
        "name": "Groq / llama-3.3-70b",
        "loader": lambda: ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0.3,
        ),
    },
    {
        "name": "Gemini / gemini-2.5-flash",
        "loader": lambda: ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ["GEMINI_API_KEY"],
            temperature=0.3,
        ),
    },
    {
        "name": "OpenRouter / llama-3.3-70b",
        "loader": lambda: ChatOpenAI(
            model="meta-llama/llama-3.3-70b-instruct:free",
            openai_api_key=os.environ["OPEN_ROUTER_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3,
            default_headers={
                "HTTP-Referer": "https://herbsaremyworld.com",
                "X-Title": "Herbal LLM Chatbot",
            },
        ),
    },
    {
        "name": "OpenRouter / deepseek-r1",
        "loader": lambda: ChatOpenAI(
            model="deepseek/deepseek-r1:free",
            openai_api_key=os.environ["OPEN_ROUTER_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3,
            default_headers={
                "HTTP-Referer": "https://herbsaremyworld.com",
                "X-Title": "Herbal LLM Chatbot",
            },
        ),
    },
]


def _fallback_invoke(input):
    last_error = None
    for provider in PROVIDERS:
        try:
            llm = provider["loader"]()
            response = llm.invoke(input)
            logger.info(f"✅ Used: {provider['name']}")
            return response
        except Exception as e:
            logger.warning(f"⚠️  {provider['name']} failed: {type(e).__name__}: {e}")
            last_error = e
            continue
    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


def load_llm():
    return RunnableLambda(_fallback_invoke)