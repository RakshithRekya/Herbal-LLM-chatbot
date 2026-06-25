from src.retrieval.retriever import load_retriever

r = load_retriever()

queries = [
    'herbs for sleep',
    'lavender properties',
    'chamomile uses',
    'basil medicinal',
    'mint digestion',
]

for query in queries:
    docs = r.invoke(query)
    print(f"\nQuery: '{query}'")
    print(f"  Doc 1: {docs[0].page_content[:120] if docs else 'NONE'}")
    print(f"  Doc 2: {docs[1].page_content[:120] if len(docs)>1 else 'NONE'}")
    all_boilerplate = all('Combination of herbs' in d.page_content for d in docs)
    print(f"  All boilerplate: {all_boilerplate}")