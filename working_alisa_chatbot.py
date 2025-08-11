import os
import streamlit as st
from dotenv import load_dotenv
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI  

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Alisa Sikelianos-Carter Chatbot", page_icon="🎨")

st.title("Ask Alisa Sikelianos-Carter")
st.write("Chat with an AI trained on Alisa's interviews and biographical information")

@st.cache_data
def load_and_process_documents():
    """Load artist data and split into chunks - shows LangChain text processing"""
    
    # Load documents
    all_text = ""
    files_loaded = []
    
    if os.path.exists("artist_data"):
        for filename in os.listdir("artist_data"):
            if filename.endswith(".txt"):
                try:
                    with open(f"artist_data/{filename}", "r", encoding='utf-8') as f:
                        content = f.read()
                        all_text += f"\n\n=== {filename} ===\n{content}"
                        files_loaded.append(filename)
                except Exception as e:
                    st.error(f"Error reading {filename}: {e}")
    
    if not all_text:
        return None, []
    
    # Use LangChain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(all_text)
    
    return chunks, files_loaded

@st.cache_resource
def setup_langchain_qa():
    """Create LangChain Q&A system"""
    
    # LangChain prompt template
    prompt_template = """You are an expert on artist Alisa Sikelianos-Carter. Use the following context to answer questions about her work, artistic process, philosophy, and background.

Context: {context}

Question: {question}

Answer in a conversational and insightful way:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # ChatOpenAI with gpt-3.5-turbo 
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  
        temperature=0.3,
        max_tokens=500
    )
    
    # Create chain
    qa_chain = LLMChain(llm=llm, prompt=prompt)
    
    return qa_chain

def find_relevant_chunks(chunks, question, max_chunks=3):
    """Simple relevance scoring - could be upgraded to embeddings later"""
    import re
    
    # Simple keyword matching for relevance
    question_words = set(re.findall(r'\w+', question.lower()))
    
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        overlap = len(question_words.intersection(chunk_words))
        chunk_scores.append((overlap, i, chunk))
    
    # Sort by relevance and return top chunks
    chunk_scores.sort(reverse=True)
    return [chunk for _, _, chunk in chunk_scores[:max_chunks]]

# Load data
chunks, files_loaded = load_and_process_documents()

if not chunks:
    st.error("No artist data found. Add .txt files to artist_data/ folder")
    st.stop()

# Show loaded data
with st.expander("📊 Loaded Data"):
    st.write(f"Loaded {len(files_loaded)} files: {', '.join(files_loaded)}")
    st.write(f"Created {len(chunks)} text chunks using LangChain text splitter")

# Setup LangChain
qa_chain = setup_langchain_qa()

# Initialize chat
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm here to answer questions about Alisa Sikelianos-Carter's work. I use LangChain for text processing and conversation management. What would you like to know?"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about Alisa's work, process, or philosophy..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response using LangChain
    with st.chat_message("assistant"):
        try:
            with st.spinner("Processing with LangChain..."):
                # Find relevant context
                relevant_chunks = find_relevant_chunks(chunks, prompt)
                context = "\n\n".join(relevant_chunks)
                
                # Use LangChain to generate response
                response = qa_chain.run(context=context[:4000], question=prompt)
                
                st.markdown(response)
                
                # Show sources
                with st.expander("📚 Context Used"):
                    st.write(f"Used {len(relevant_chunks)} most relevant text chunks")
                    for i, chunk in enumerate(relevant_chunks[:2]):
                        st.write(f"**Chunk {i+1}:** {chunk[:300]}...")
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            # Show more details for debugging
            import traceback
            st.error(traceback.format_exc())

# Sidebar
with st.sidebar:
    st.subheader("💡 Try asking:")
    example_questions = [
        "What is Alisa's artistic philosophy?",
        "How does she approach her creative process?",
        "What themes does she explore?",
        "What influences her work?",
        "Tell me about her background"
    ]
    
    for question in example_questions:
        if st.button(question):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    
    st.markdown("---")
    st.subheader("🔧 LangChain Features Used:")
    st.write("- Text Splitter")
    st.write("- Prompt Templates") 
    st.write("- LLM Chain")
    st.write("- Chat Models")
    st.write("- Conversation Management")