import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from ddgs import DDGS

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_CHOICES = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]

USE_QUERY_REWRITE = False

STRICT_PROMPT = """You are a precise document assistant. Follow these rules:

1. Use only information found in the context chunks. You may combine several chunks, which is important for broad questions like "what is this document about".
2. Only say "I couldn't find that in the document." if the context has nothing relevant at all.
3. Never invent facts and never use outside knowledge.
4. For specific questions, quote or closely paraphrase the relevant chunk.
5. Be concise."""

HYBRID_PROMPT = """You are a helpful document assistant. You receive (1) context chunks from the user's document and (2) web search results, which may be empty.

Rules:
1. Start from the document context. Combine several chunks when the question is broad.
2. If the question needs something the document does not say (who organises it, background, definitions, examples, how-to), use the web search results and your general knowledge. Do not refuse just because the PDF is silent.
3. When both sources are used, label them exactly: "📄 From your document:" and "🌐 From the web / general knowledge:"
4. Never present web or general knowledge as if the document said it. If the web results do not confirm something, say you are not sure instead of guessing.
5. For examples, give clear ones that fit the document's topic.
6. Be concise. Use short bullet points for lists."""

IMAGE_PROMPT = """You are a helpful assistant. You are shown an image of ONE page from a document, and possibly some web search results.

Rules:
1. Answer using what you can see in the page image first.
2. If the page does not have enough information and web results are given, use them too, and label that part "🌐 From the web:"
3. If you are not confident, say so instead of guessing.
4. Be concise."""


def call_gemini_multimodal(parts, system_prompt):
    config = types.GenerateContentConfig(system_instruction=system_prompt)
    last_error = None

    for model_name in MODEL_CHOICES:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name, contents=parts, config=config
                )
                print("Answered by:", model_name)
                return response.text or "I couldn't produce an answer. Try rephrasing."
            except Exception as e:
                last_error = e
                error_text = str(e)
                if "503" in error_text or "UNAVAILABLE" in error_text:
                    time.sleep(4)
                    continue
                break

    raise last_error


def answer_about_page(image_bytes, question, use_web=True, doc_hint="", mime_type="image/png"):
    parts = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        question,
    ]
    sources = []

    if use_web:
        query = (question + " " + doc_hint[:70]).strip()
        web_results = search_web(query)
        web_text = ""
        for title, link, snippet in web_results:
            web_text = web_text + "- " + title + ": " + snippet + "\n"
            sources.append((title or link, link))
        if web_text:
            parts.append("Web search results:\n" + web_text)

    try:
        answer = call_gemini_multimodal(parts, IMAGE_PROMPT)
        return answer, sources
    except Exception as e:
        return friendly_error(e), []

def call_gemini(prompt, system_prompt):
    config = types.GenerateContentConfig(system_instruction=system_prompt)
    last_error = None

    for model_name in MODEL_CHOICES:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name, contents=prompt, config=config
                )
                print("Answered by:", model_name)
                return response.text or "I couldn't produce an answer. Try rephrasing."
            except Exception as e:
                last_error = e
                error_text = str(e)
                if "503" in error_text or "UNAVAILABLE" in error_text:
                    time.sleep(4)
                    continue
                break

    raise last_error


def make_search_query(question, retrieved_chunks):
    topic = ""
    if retrieved_chunks:
        topic = retrieved_chunks[0][:200].replace("\n", " ")

    ask = (
        "Write ONE short web search query (max 10 words) that would find the answer "
        "to the question below. Use distinctive names from the document excerpt "
        "(event, company, challenge or task names).\n\n"
        "Document excerpt: " + topic + "\n\n"
        "Question: " + question + "\n\n"
        "Reply with only the query."
    )
    try:
        query = call_gemini(ask, "You write concise web search queries.")
        return query.strip().strip('"')[:120]
    except Exception:
        return (question + " " + topic[:60])[:120]


def search_web(query, max_results=5):
    try:
        results = DDGS(timeout=6).text(query, max_results=max_results)
    except Exception as e:
        print("Web search error:", e)
        return []

    found = []
    for r in results or []:
        found.append((r.get("title", ""), r.get("href", ""), r.get("body", "")))
    return found


def friendly_error(error):
    error_text = str(error)
    print("Gemini error:", error_text)
    if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
        return "⚠️ The free usage limit was reached for now. Please try again in a few minutes."
    return "Sorry, something went wrong: " + error_text[:200]


def answer_question(question, retrieved_chunks, use_web=True, doc_hint=""):
    start = time.perf_counter()
    context = "\n\n---\n\n".join(retrieved_chunks)

    if not use_web:
        prompt = "Context from the document:\n" + context + "\n\nQuestion: " + question
        try:
            answer = call_gemini(prompt, STRICT_PROMPT)
            print("Timing: Gemini answer", round(time.perf_counter() - start, 1), "s")
            return answer, []
        except Exception as e:
            return friendly_error(e), []

    if USE_QUERY_REWRITE:
        query = make_search_query(question, retrieved_chunks)
    else:
        query = (question + " " + doc_hint[:70]).strip()
    print("Web search query:", query)
    print("Timing: query ready", round(time.perf_counter() - start, 1), "s")

    step = time.perf_counter()
    web_results = search_web(query)
    print("Timing: web search", round(time.perf_counter() - step, 1), "s")

    web_text = ""
    sources = []
    for title, link, snippet in web_results:
        web_text = web_text + "- " + title + ": " + snippet + "\n"
        sources.append((title or link, link))
    if not web_text:
        web_text = "(no web results were found)"

    prompt = (
        "Context from the document:\n" + context
        + "\n\nWeb search results:\n" + web_text
        + "\n\nQuestion: " + question
    )

    step = time.perf_counter()
    try:
        answer = call_gemini(prompt, HYBRID_PROMPT)
        print("Timing: Gemini answer", round(time.perf_counter() - step, 1), "s")
        return answer, sources
    except Exception as e:
        return friendly_error(e), []


if __name__ == "__main__":
    test_chunks = ["The sky appears blue because of Rayleigh scattering."]
    answer, sources = answer_question("Why is the sky blue?", test_chunks, use_web=False)
    print("Answer:", answer)