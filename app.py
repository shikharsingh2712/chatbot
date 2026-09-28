import streamlit as st

from document_loader import load_pdf_text
from chunker import chunk_text
from vectorstore import VectorStore
from qa_engine import answer_question  
from qa_engine import answer_about_page


st.set_page_config(page_title="Document Q&A Chatbot", page_icon="📄")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, p, label, [data-testid="stMarkdownContainer"], 
    [data-testid="stChatMessage"], button, input, textarea {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    }

    .app-header .title-text h1 {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stMainBlockContainer"], .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 6rem !important;
        max-width: 800px;
    }
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .app-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 0 0 20px 0;
    }
    .app-header .logo {
        font-size: 30px;
        background: linear-gradient(135deg, #7C3AED, #3B82F6);
        width: 55px;
        height: 55px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .app-header .title-text h1 {
        margin: 0;
        padding: 0;
        font-size: 26px;
    }
    .app-header .title-text p {
        margin: 0;
        color: gray;
        font-size: 14px;
    }

    .greeting {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(124, 58, 237, 0.35);
        padding: 10px 14px;
        border-radius: 12px;
        margin-bottom: 18px;
        font-size: 14px;
    }
    .greeting-avatar {
        font-size: 20px;
        background: linear-gradient(135deg, #7C3AED, #3B82F6);
        width: 34px;
        height: 34px;
        min-width: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    @media (max-width: 600px) {
        .app-header { gap: 10px; }
        .app-header .logo { width: 42px; height: 42px; font-size: 22px; }
        .app-header .title-text h1 { font-size: 20px; }
        .app-header .title-text p { font-size: 12px; }
        .greeting { font-size: 13px; padding: 8px 10px; }
        [data-testid="stChatMessage"] { font-size: 14px; padding: 10px 12px; }
        .custom-footer { font-size: 13px; padding: 10px; }
        .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
    }

    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        padding: 14px;
        font-size: 16px;
        font-weight: 600;
        color: white;
        background: linear-gradient(135deg, #7C3AED, #3B82F6);
        z-index: 100;
    }
</style>

<div class="app-header">
    <div class="logo">📄</div>
    <div class="title-text">
        <h1>Document Q&A Chatbot</h1>
        <p>Ask questions about your PDF, grounded in it — with optional web lookup.</p>
    </div>
</div>
<div class="greeting">
    <span class="greeting-avatar">🙋</span>
    <span>Hey there! Upload a PDF below and ask me anything — I'll answer from the document, and can search the web for anything it doesn't cover.</span>
</div>

<div class="custom-footer">✨ Built by Shikhar Shankar Singh</div>
""",
    unsafe_allow_html=True,
)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "focus_index" not in st.session_state:
    st.session_state.focus_index = None
if "doc_key" not in st.session_state:
    st.session_state.doc_key = None
if "doc_hint" not in st.session_state:
    st.session_state.doc_hint = ""


def shorten(text, limit=32):
    text = text.replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def show_message(msg):
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        sources = msg.get("sources", [])
        if sources:
            with st.expander("🌐 Web sources"):
                for title, uri in sources:
                    st.markdown(f"- [{title}]({uri})")

        with st.expander("📋 Copy this text"):
            st.code(msg["content"], language=None, wrap_lines=True)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    file_key = f"{uploaded_file.name}-{uploaded_file.size}"

    if st.session_state.doc_key != file_key:

        with st.spinner("📖 Reading your PDF..."):
            text = load_pdf_text(uploaded_file)
        
        

        if not text.strip():
            st.error(
                "No readable text found. This PDF is probably scanned images. "
                "It needs OCR first (a good future upgrade)."
            )
            st.stop()

        chunks = chunk_text(text)

        progress = st.progress(0, text="🧠 Understanding your document...")

        def update_progress(done, total):
            progress.progress(
                done / total, text=f"🧠 Understanding your document... {done}/{total}"
            )

        store = VectorStore()
        store.build_index(chunks, progress_callback=update_progress)
        progress.empty()

        st.session_state.vector_store = store
        st.session_state.doc_key = file_key
        st.session_state.doc_hint = " ".join(text[:150].split())
        st.session_state.messages = []
        st.session_state.focus_index = None
        st.success(f"✅ Ready! Indexed {len(chunks)} pieces. Ask your questions below.")

elif st.session_state.vector_store is not None:
    st.session_state.vector_store = None
    st.session_state.doc_key = None
    st.session_state.messages = []
    st.session_state.focus_index = None

with st.sidebar:
    st.header("⚙️ Settings")
    use_web = st.toggle(
        "🌐 Use web knowledge",
        value=True,
        help="ON: the bot may search Google for things the PDF does not say "
        "(who organises it, examples, background). "
        "OFF: strict mode, answers only from the PDF.",
    )

    st.divider()
    st.header("💬 History")

    user_questions = [
        (index, msg["content"])
        for index, msg in enumerate(st.session_state.messages)
        if msg["role"] == "user"
    ]

    if not user_questions:
        st.caption("Your questions will show up here. Click one to jump back to it.")
    else:
        for number, (index, question_text) in enumerate(user_questions, start=1):
            label = f"Q{number}: {shorten(question_text)}"
            if st.button(label, key=f"history_{index}", use_container_width=True):
                st.session_state.focus_index = index

        if st.session_state.focus_index is not None:
            if st.button("⬅ Back to full chat", type="primary", use_container_width=True):
                st.session_state.focus_index = None

messages = st.session_state.messages
focus = st.session_state.focus_index

if focus is not None and focus < len(messages):
    st.info("📌 Showing the question you picked. Use “Back to full chat” in the sidebar.")
    show_message(messages[focus])
    if focus + 1 < len(messages):
        show_message(messages[focus + 1])
else:
    for msg in messages:
        show_message(msg)


    with st.expander("🖼️ Ask about an image"):
        uploaded_image = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg"], key="image_uploader"
        )
        image_question = st.text_input("Your question about this image", key="image_question")

        if uploaded_image is not None and image_question and st.button("Ask about this image"):
            image_bytes = uploaded_image.getvalue()
            st.image(image_bytes, caption=uploaded_image.name)

            with st.spinner("Looking at the image..."):
                image_answer, image_sources = answer_about_page(
                    image_bytes,
                    image_question,
                    use_web=use_web,
                    doc_hint=st.session_state.doc_hint,
                    mime_type=uploaded_image.type,
                )
            st.markdown(image_answer)

            if image_sources:
                with st.expander("🌐 Web sources"):
                    for title, uri in image_sources:
                        st.markdown(f"- [{title}]({uri})")

if st.session_state.vector_store is not None:
    question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.focus_index = None
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                relevant_chunks = st.session_state.vector_store.search(question, top_k=10)
                answer, sources = answer_question(
                    question, relevant_chunks, use_web=use_web,
                    doc_hint=st.session_state.doc_hint,
                )
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.rerun()