from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retrieval.retriever import load_retriever
from src.llm.model import load_llm
from src.ingest.translator import is_greek

PROMPT_TEMPLATE = """
You are a knowledgeable herbal assistant for a herbal company.
Use ONLY the context below to answer the question accurately.
Be specific and direct. Use herb names when mentioned in the context.
If the answer is not in the context, say: "I don't have information on that in my current knowledge base. Please consult a specialist."
Do NOT add extra commentary, suggestions, or filler phrases.

LANGUAGE RULE: You MUST respond in the same language as the question. No exceptions.
- Question in English → answer in English only
- Question in Greek → answer in Greek only

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
