import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document

load_dotenv()

st.set_page_config(page_title="Alisa Sikelianos-Carter Chatbot", page_icon="🎨")
st.title("🎨 Ask Alisa Sikelianos-Carter")
st.write("AI chatbot using LangChain for document processing and conversation management")

def get_openai_key():
    """Get OpenAI API key with proper validation"""
    api_key = None
    
    if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success("🔑 API key loaded from Streamlit secrets")
    
    elif os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")
        st.success("🔑 API key loaded from environment")
    
    if not api_key:
        st.error("❌ OpenAI API key not found!")
        st.info("Please add your API key to Streamlit secrets or .env file")
        return None
    
    if not api_key.startswith("sk-"):
        st.error("❌ Invalid API key format (should start with 'sk-')")
        return None
    
    if len(api_key) < 20:
        st.error("❌ API key seems too short")
        return None
    
    return api_key

def validate_openai_connection(api_key):
    """Test OpenAI connection with older API"""
    try:
        import openai
        
        openai.api_key = api_key
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)

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

def create_qa_system():
    """Create QA system using LangChain prompt templates and OpenAI 0.28.1"""
    
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
    
    class QASystem:
        def __init__(self, prompt_template):
            self.prompt_template = prompt_template
        
        def answer(self, context, question):
            import openai
            
            # Use LangChain prompt template
            formatted_prompt = self.prompt_template.format(
                context=context, 
                question=question
            )
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": formatted_prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
    
    return QASystem(prompt)

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

openai_api_key = get_openai_key()

if not openai_api_key:
    st.stop()

with st.spinner("Testing OpenAI connection..."):
    connection_ok, message = validate_openai_connection(openai_api_key)
    
    if connection_ok:
        st.success(f"✅ OpenAI connection successful!")
    else:
        st.error(f"❌ OpenAI connection failed: {message}")
        st.stop()

# Load and process documents using LangChain
with st.spinner("Processing documents with LangChain..."):
    documents, files_loaded = load_and_process_documents()

qa_system = create_qa_system()

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

st.success("- System ready!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hi! I use LangChain for document processing and prompt management. I'm ready to answer questions about Alisa Sikelianos-Carter's work and artistic practice. What would you like to know?"
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

    # Generate response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Processing with LangChain..."):
                # Use LangChain retrieval
                relevant_docs = find_relevant_chunks(documents, prompt)
                
                # Combine context
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Get answer using LangChain prompt + OpenAI
                response = qa_system.answer(context[:4000], prompt)
                
                st.markdown(response)
                
                # Show sources
                with st.expander("📚 Sources Used"):
                    for i, doc in enumerate(relevant_docs):
                        st.write(f"**{i+1}.** {doc.metadata.get('source', 'Unknown')}")
                        st.write(f"_{doc.page_content[:150]}..._")
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

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
    st.subheader("🔧 LangChain Features:")
    st.write("- Document Processing")
    st.write("- Text Splitters")
    st.write("- Prompt Templates") 
    st.write("- Schema Objects")
    st.write("- Retrieval System")
    
    st.markdown("---")
    st.info("Using OpenAI 0.28.1 for compatibility")
    st.write(f"**Documents:** {len(documents)} chunks")

st.markdown("---")
st.markdown("*LangChain + OpenAI + Streamlit Integration*")