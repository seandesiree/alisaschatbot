import os
import pathlib
import anthropic
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

st.set_page_config(
    page_title="Ask Alisa Sikelianos-Carter",
    page_icon="✦",
    layout="centered",
)

# ── Design ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=Jost:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, .stApp {
    background-color: #07070d;
    color: #e0d8cc;
}

* {
    font-family: 'Jost', sans-serif;
    font-weight: 300;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0a0a14 !important;
    border-right: 1px solid rgba(196, 169, 106, 0.12);
}

[data-testid="stSidebar"] * {
    color: #b0a898 !important;
}

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300 !important;
    letter-spacing: 0.06em !important;
    color: #e8e0d0 !important;
}

/* ── Custom header block ── */
.alisa-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(196, 169, 106, 0.18);
    margin-bottom: 1.5rem;
}

.alisa-header .symbol {
    font-size: 1.6rem;
    color: #c4a96a;
    letter-spacing: 0.4em;
    display: block;
    margin-bottom: 0.6rem;
    opacity: 0.8;
}

.alisa-header h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 3.4rem !important;
    font-weight: 300 !important;
    color: #ede5d5 !important;
    letter-spacing: 0.08em !important;
    margin: 0 0 0.5rem 0 !important;
    line-height: 1.2 !important;
}

.alisa-header .subtitle {
    font-family: 'Jost', sans-serif;
    font-size: 0.95rem;
    font-weight: 300;
    color: #7a7268;
    letter-spacing: 0.25em;
    text-transform: uppercase;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: rgba(255, 255, 255, 0.025) !important;
    border: 1px solid rgba(196, 169, 106, 0.1) !important;
    border-radius: 2px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.75rem !important;
}

[data-testid="stChatMessage"] p {
    color: #d8d0c0 !important;
    font-size: 1.1rem !important;
    line-height: 1.85 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    border-top: 1px solid rgba(196, 169, 106, 0.15) !important;
    padding-top: 1rem !important;
}

[data-testid="stChatInputTextArea"] {
    background-color: #0c0c18 !important;
    color: #e0d8cc !important;
    border: 1px solid rgba(196, 169, 106, 0.25) !important;
    border-radius: 2px !important;
    font-family: 'Jost', sans-serif !important;
    font-weight: 300 !important;
    font-size: 1.1rem !important;
}

[data-testid="stChatInputTextArea"]:focus {
    border-color: rgba(196, 169, 106, 0.5) !important;
    box-shadow: 0 0 0 1px rgba(196, 169, 106, 0.15) !important;
}

/* ── Sidebar buttons ── */
.stButton > button {
    background-color: transparent !important;
    color: #a09080 !important;
    border: 1px solid rgba(196, 169, 106, 0.2) !important;
    border-radius: 1px !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.06em !important;
    padding: 0.4rem 0.8rem !important;
    width: 100% !important;
    text-align: left !important;
    transition: all 0.25s ease !important;
    margin-bottom: 0.3rem !important;
}

.stButton > button:hover {
    background-color: rgba(196, 169, 106, 0.07) !important;
    color: #c4a96a !important;
    border-color: rgba(196, 169, 106, 0.4) !important;
}

/* ── Expander (sources) ── */
[data-testid="stExpander"] {
    background-color: rgba(255, 255, 255, 0.015) !important;
    border: 1px solid rgba(196, 169, 106, 0.1) !important;
    border-radius: 1px !important;
}

[data-testid="stExpander"] summary {
    color: #6a6258 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #6a6258 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(196, 169, 106, 0.15) !important;
    margin: 1.5rem 0 !important;
}

/* ── Footer ── */
.alisa-footer {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
    color: #3a3830;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}

/* ── Sidebar heading ── */
.sidebar-heading {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    font-weight: 300;
    color: #7a7060 !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(196, 169, 106, 0.12);
}

.sidebar-meta {
    font-size: 0.8rem;
    color: #3a3830 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 1.5rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #07070d; }
::-webkit-scrollbar-thumb { background: rgba(196, 169, 106, 0.2); border-radius: 2px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="alisa-header">
    <span class="symbol">✦ &nbsp; ✦ &nbsp; ✦</span>
    <h1>Alisa Sikelianos-Carter</h1>
    <p class="subtitle">Ask about her work, practice &amp; world</p>
</div>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_anthropic_key():
    api_key = None

    secrets_paths = [
        pathlib.Path.home() / ".streamlit" / "secrets.toml",
        pathlib.Path(__file__).parent / ".streamlit" / "secrets.toml",
    ]
    if any(p.exists() for p in secrets_paths) and "ANTHROPIC_API_KEY" in st.secrets:
        api_key = st.secrets["ANTHROPIC_API_KEY"]

    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        st.error("ANTHROPIC_API_KEY not found. Add it to your .env file or Streamlit secrets.")
        return None

    return api_key


def validate_anthropic_connection(api_key):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        client.messages.create(
            model="claude-opus-4-7",
            max_tokens=5,
            messages=[{"role": "user", "content": "Hi"}],
        )
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)


@st.cache_data
def load_and_process_documents():
    documents = []
    files_loaded = []

    if os.path.exists("artist_data"):
        for filename in sorted(os.listdir("artist_data")):
            if filename.endswith(".txt"):
                try:
                    filepath = os.path.join("artist_data", filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            doc = Document(
                                page_content=content,
                                metadata={"source": filename, "type": "artist_info"},
                            )
                            documents.append(doc)
                            files_loaded.append(filename)
                except Exception as e:
                    st.warning(f"Could not load {filename}: {e}")

    if not documents:
        documents = [
            Document(
                page_content="Alisa Sikelianos-Carter is a Black, Queer mixed-media artist from upstate New York. Her practice is grounded in ancestral reverence, intuitive research, and visual theology.",
                metadata={"source": "bio.txt", "type": "biography"},
            )
        ]
        files_loaded = ["demo_bio.txt"]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_documents(documents), files_loaded


def find_relevant_chunks(documents, query, max_chunks=3):
    import re
    query_words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for doc in documents:
        doc_words = set(re.findall(r"\w+", doc.page_content.lower()))
        scored.append((len(query_words & doc_words), doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:max_chunks]]


def stream_answer(client, context, question):
    system_prompt = f"""You are a thoughtful assistant specializing in the work and artistic practice of Alisa Sikelianos-Carter — a Black, Queer mixed-media artist whose practice is grounded in ancestral reverence, visual theology, and the exploration of loss, shadow work, and mythopoetics.

Use the context below to answer questions about her work, artistic process, philosophy, influences, and background. Be accurate, insightful, and speak with the same care and intention that characterizes her practice.

Context:
{context}"""

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# ── Initialization ───────────────────────────────────────────────────────────

anthropic_api_key = get_anthropic_key()
if not anthropic_api_key:
    st.stop()

with st.spinner(""):
    connection_ok, message = validate_anthropic_connection(anthropic_api_key)
    if not connection_ok:
        st.error(f"Connection failed: {message}")
        st.stop()

anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

with st.spinner(""):
    documents, files_loaded = load_and_process_documents()


# ── Chat ─────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "I'm here to guide you through the work and world of Alisa Sikelianos-Carter. What would you like to explore?",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            relevant_docs = find_relevant_chunks(documents, prompt)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            placeholder = st.empty()
            response_text = ""
            for chunk in stream_answer(anthropic_client, context[:4000], prompt):
                response_text += chunk
                placeholder.markdown(response_text + "▌")
            placeholder.markdown(response_text)
            with st.expander("sources"):
                for i, doc in enumerate(relevant_docs):
                    st.write(f"**{i + 1}.** {doc.metadata.get('source', 'Unknown')}")
                    st.write(f"_{doc.page_content[:150]}..._")
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="sidebar-heading">Enter here</p>', unsafe_allow_html=True)

    example_questions = [
        "What is Alisa's artistic philosophy?",
        "How does she approach her creative process?",
        "What themes does she explore?",
        "What influences her work?",
        "Tell me about her background",
    ]

    for question in example_questions:
        if st.button(question, key=f"q_{hash(question)}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

    st.markdown(
        f'<p class="sidebar-meta">{len(documents)} passages loaded</p>',
        unsafe_allow_html=True,
    )


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown('<div class="alisa-footer">✦</div>', unsafe_allow_html=True)
