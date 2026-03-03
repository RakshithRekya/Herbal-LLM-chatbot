from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retrieval.retriever import load_retriever
from src.llm.model import load_llm

PROMPT_TEMPLATE = """
You are a knowledgeable herbal assistant for a herbal company.
Use the context below to answer the question accurately.
If the answer is not in the context, say you don't have enough information on that topic.

Important instructions:
- Detect the language of the user's question automatically.
- Always respond in the same language as the question.
- If the question is in Greek, answer in Greek.
- If the question is in English, answer in English.
- Do not translate the question, just answer in the same language.

Context:
{context}

Question:
{question}

Answer:
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain():
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    retriever = load_retriever()
    llm = load_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
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
        answer = chain.invoke(question)
        print(f"\nAssistant: {answer}\n")

if __name__ == "__main__":
    chat()
