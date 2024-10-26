import os
import logging
import streamlit as st
from rag import RAGSystem, DataBaseCollector

# Set environment variable to avoid OpenMP errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def initialize_session_state():
    """Initialize session state variables."""
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

def initialize_rag_system():
    """Initialize RAG system components."""
    collector = DataBaseCollector(
        host="localhost",
        port=27017,
        database="confluence",
        collection_name="pages",
        context_size=200
    )
    
    collector.update([163978, 131097])

    rag_system = RAGSystem(
        faiss_index_path="faiss.index",
        # mongo_host="localhost",
        # mongo_port=27017,
        mongo_db="confluence",
        mongo_collection="pages"
    )
    
    return rag_system

def handle_user_input(rag_system, user_input: str):
    """Process user input and generate response."""
    if user_input:
        try:
            # Generate response
            response = rag_system.generate_response(user_input)
            
            # Update conversation memory
            st.session_state.conversation_memory.append((user_input, response))
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            st.error("An error occurred while generating the response.")

def display_chat_history():
    """Display chat messages using Streamlit's chat message containers."""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user", avatar="🧑").markdown(message["content"])
        else:
            st.chat_message("assistant", avatar="🤖").markdown(message["content"])

def main():
    # first update the database
    rag_system = initialize_rag_system()

    st.title("RAG System Chat Interface")

    # Initialize session state and RAG system
    initialize_session_state()

    # Display chat history
    display_chat_history()

    # Chat input
    if user_input := st.chat_input("Type your question here..."):
        handle_user_input(rag_system, user_input)
        st.rerun()

if __name__ == "__main__":
    main()