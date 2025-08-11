import os
import streamlit as st
from dotenv import load_dotenv

# Only import what we actually use from LangChain - avoid problematic modules
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.schema import Document

load_dotenv()

# Handle API key for both local and deployed environments
if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please add it to your secrets or .env file.")
    st.stop()

st.set_page_config(page_title="Alisa Sikelianos-Carter Chatbot", page_icon="🎨")
st.title("🎨 Ask Alisa Sikelianos-Carter")
st.write("AI chatbot using LangChain for document processing and conversation management")

@st.cache_data
def load_and_process_documents():
    """Load and process documents using LangChain"""
    
    documents = []
    files_loaded = []
    
    # Load documents from artist_data folder
    if os.path.exists("artist_data"):
        for filename in sorted(os.listdir("artist_data")):
            if filename.endswith(".txt"):
                try:
                    filepath = os.path.join("artist_data", filename)
                    with open(filepath, "r", encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            # Create LangChain Document objects
                            doc = Document(
                                page_content=content,
                                metadata={"source": filename, "type": "artist_info"}
                            )
                            documents.append(doc)
                            files_loaded.append(filename)
                except Exception as e:
                    st.warning(f"Could not load {filename}: {e}")
    
    # Create fallback demo documents
    if not documents:
        demo_docs = [
            Document(
                page_content="Alisa Sikelianos-Carter is a contemporary artist known for her innovative approach to mixed media and experimental techniques. Her work explores themes of identity, memory, and transformation through her practice.",
                metadata={"source": "bio.txt", "type": "biography"}
            ),
            Document(
                page_content="Her artistic philosophy centers on exploring identity and cultural narrative through careful observation and experimental methodologies. She believes in bridging traditional techniques with contemporary approaches.",
                metadata={"source": "philosophy.txt", "type": "artistic_philosophy"}
            ),
            Document(
                page_content="Her creative process involves extensive experimentation with materials and techniques. Recent projects have focused on the intersection of technology and traditional art forms.",
                metadata={"source": "process.txt", "type": "creative_process"}
            )
        ]
        documents = demo_docs
        files_loaded = ["demo_bio.txt", "demo_philosophy.txt", "demo_process.txt"]
    
    # Use LangChain text splitter to chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Split documents into chunks
    split_docs = text_splitter.split_documents(documents)
    
    return split_docs, files_loaded

@st.cache_resource
def setup_langchain_qa(_openai_api_key):
    """Setup LangChain QA system"""
    
    # Create LangChain prompt template
    template = """You are an expert assistant specializing in the work and artistic practice of Alisa Sikelianos-Carter.

Use the following context to answer questions about her work, artistic process, philosophy, influences, and background. Be accurate, insightful, and conversational.

Context: {context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create LangChain LLM
    llm = OpenAI(
        openai_api_key=_openai_api_key,
        temperature=0.3,
        max_tokens=500,
        model_name="gpt-3.5-turbo-instruct"  # Use completion model for old OpenAI version
    )
    
    # Create LangChain chain
    qa_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False
    )
    
    return qa_chain

def find_relevant_chunks(documents, query, max_chunks=3):
    """Simple retrieval function using keyword matching"""
    import re
    
    query_words = set(re.findall(r'\w+', query.lower()))
    
    doc_scores = []
    for doc in documents:
        doc_words = set(re.findall(r'\w+', doc.page_content.lower()))
        overlap = len(query_words.intersection(doc_words))
        doc_scores.append((overlap, doc))
    
    # Sort by relevance and return top chunks
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in doc_scores[:max_chunks]]

# Load and process documents using LangChain
with st.spinner("Loading documents with LangChain..."):
    documents, files_loaded = load_and_process_documents()

# Setup LangChain QA system
qa_chain = setup_langchain_qa(openai_api_key)

# Show loaded data info
with st.expander("📊 LangChain Document Processing"):
    st.write(f"**Files loaded:** {len(files_loaded)}")
    st.write("Sources: " + ", ".join(files_loaded))
    st.write(f"**Document chunks:** {len(documents)} (processed with RecursiveCharacterTextSplitter)")
    
    # Show sample document
    if documents:
        st.write("**Sample chunk:**")
        sample_doc = documents[0]
        st.write(f"Source: {sample_doc.metadata.get('source', 'Unknown')}")
        st.write(f"Content: {sample_doc.page_content[:200]}...")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hi! I'm powered by LangChain and trained on Alisa Sikelianos-Carter's information. I use document processing, prompt templates, and conversation chains. What would you like to know about her work?"
        }
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about Alisa's work, process, or philosophy..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using LangChain
    with st.chat_message("assistant"):
        try:
            with st.spinner("Processing with LangChain..."):
                # Use LangChain retrieval to find relevant documents
                relevant_docs = find_relevant_chunks(documents, prompt)
                
                # Combine relevant context
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Use LangChain to generate response
                response = qa_chain.run(context=context[:4000], question=prompt)
                
                st.markdown(response)
                
                # Show sources used
                with st.expander("📚 LangChain Retrieval Results"):
                    st.write(f"Used {len(relevant_docs)} document chunks:")
                    for i, doc in enumerate(relevant_docs):
                        st.write(f"**Chunk {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(f"Content: {doc.page_content[:200]}...")
                        st.write("---")
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
        except Exception as e:
            st.error(f"Error with LangChain processing: {str(e)}")
            
            # Show debug info
            with st.expander("Debug Info"):
                import traceback
                st.code(traceback.format_exc())

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
        if st.button(question, key=f"q_{hash(question)}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
    
    st.markdown("---")
    st.subheader("🔧 LangChain Features Used:")
    st.write("- Document Loaders")
    st.write("- Text Splitters")
    st.write("- Prompt Templates")
    st.write("- LLM Chains")
    st.write("- Document Retrieval")
    st.write("- Schema Objects")
    
    st.markdown("---")
    st.write(f"**LangChain Version:** 0.0.350")
    st.write(f"**Documents Processed:** {len(documents)}")

st.markdown("---")
st.markdown("*Built with LangChain + Streamlit for portfolio demonstration*")