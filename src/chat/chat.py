from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from deep_translator import GoogleTranslator
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.retrieval.retriever import load_retriever
from src.llm.model import load_llm
from src.ingest.translator import is_greek

ANSWER_PROMPT = """
You are a knowledgeable herbal assistant.
Use ONLY the context below to answer the question.
Be specific and direct.
If the answer is not in the context, say: "I don't have information on that in my current knowledge base. Please consult a specialist."

Context:
{context}

Question asked in {language}: {question}

IMPORTANT: Your response must be written entirely in {language}. Do not use any other language.

Answer in {language}:
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def translate_to_greek(text):
    try:
        return GoogleTranslator(source='en', target='el').translate(text)
    except:
        return text

def build_chain():
    prompt = PromptTemplate(
        template=ANSWER_PROMPT,
        input_variables=["context", "question", "language"]
    )
    retriever = load_retriever()
    llm = load_llm()

    def pipeline(question):
        if not is_greek(question):
            language = "English"
            greek_question = translate_to_greek(question)
        else:
            language = "Greek"
            greek_question = question

        docs = retriever.invoke(greek_question)
        context = format_docs(docs)

        return (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": question,
            "language": language
        })

    return RunnableLambda(pipeline)

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