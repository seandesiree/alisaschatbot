# Alisa's AI Chatbot

A conversational AI chatbot trained on Alisa Sikelianos-Carter's interviews and biographical information using LangChain and OpenAI.

## Features
- Real-time chat interface
- Source citation for responses
- YouTube interview transcript integration
- Vector-based retrieval system

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your OpenAI API key to `.env`
4. Run transcript extraction: `python transcript_extractor.py`
5. Launch the app: `streamlit run alisa_chatbot.py`

## Tech Stack
- **LangChain**: RAG implementation and conversation management
- **OpenAI**: Embeddings and chat completion
- **ChromaDB**: Vector database
- **Streamlit**: Web interface
- **YouTube Transcript API**: Automatic transcript extraction