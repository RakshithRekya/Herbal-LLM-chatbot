import os
import logging
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO)
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
        "name": "OpenRouter / free-router",
        "loader": lambda: ChatOpenAI(
            model="openrouter/free",   # ← auto-selects best available free model
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
            print(f"✅ Provider used: {provider['name']}", flush=True)  # guaranteed visible in Render logs

            # Tag the response so callers (chat.py / app.py) can access it if needed
            try:
                if hasattr(response, "response_metadata"):
                    response.response_metadata["provider_used"] = provider["name"]
                else:
                    response.response_metadata = {"provider_used": provider["name"]}
            except Exception:
                pass  # non-critical if response type doesn't support metadata assignment

            return response
        except Exception as e:
            logger.warning(f"⚠️  {provider['name']} failed: {type(e).__name__}: {e}")
            print(f"⚠️  {provider['name']} failed: {type(e).__name__}: {e}", flush=True)
            last_error = e
            continue
    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


def load_llm():
    return RunnableLambda(_fallback_invoke)