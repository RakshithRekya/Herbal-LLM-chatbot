import os
os.environ["OLLAMA_NUM_GPU"] = "99"

from langchain_ollama import OllamaLLM
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import LLM_MODEL

def load_llm():
    return OllamaLLM(model=LLM_MODEL)