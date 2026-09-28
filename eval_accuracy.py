import time

from document_loader import load_pdf_text
from chunker import chunk_text
from vectorstore import VectorStore
from qa_engine import answer_question

PDF_PATH = "sample.pdf"

TESTS = [
    {"q": "Which output files must be submitted?",
     "keywords": ["matching_results.tsv", "candidate_pairs.tsv"]},
    {"q": "What kind of noise is in the data?",
     "keywords": ["name", "address"]},
    {"q": "What is the task about?",
     "keywords": ["entity"]},
]

text = load_pdf_text(PDF_PATH)
chunks = chunk_text(text)
store = VectorStore()
store.build_index(chunks)

retrieval_hits = 0
answer_hits = 0

for test in TESTS:
    found = store.search(test["q"], top_k=10)
    found_text = " ".join(found).lower()
    retrieved_ok = all(k.lower() in found_text for k in test["keywords"])

    answer, _ = answer_question(test["q"], found, use_web=False)
    answered_ok = all(k.lower() in answer.lower() for k in test["keywords"])

    retrieval_hits += retrieved_ok
    answer_hits += answered_ok

    print("Q:", test["q"])
    print("  retrieval found the info:", retrieved_ok)
    print("  answer correct:", answered_ok)
    time.sleep(3)

print()
print("Retrieval score:", retrieval_hits, "/", len(TESTS))
print("Answer score:", answer_hits, "/", len(TESTS))