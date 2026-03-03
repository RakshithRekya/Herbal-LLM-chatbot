from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retrieval.retriever import load_retriever
from src.llm.model import load_llm

PROMPT_TEMPLATE = """
You are a knowledgeable herbal assistant for a herbal company.
Use the context below to answer the question accurately.
If the answer is not in the context, say "I don't have enough information on that."

Context:
{context}

Question:
{question}

Answer:
"""

def build_chain():
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        retriever=load_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

def chat():
    print("Herbal Assistant ready. Type 'exit' to quit.\n")
    chain = build_chain()
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        answer = chain.invoke({"query": question})
        print(f"\nAssistant: {answer['result']}\n")

if __name__ == "__main__":
    chat()