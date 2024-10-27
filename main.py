import os
import logging
import streamlit as st
from rag import RAGSystem, DataBaseCollector
import time

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
    """Initialize RAG system components if not already in session_state."""
    if "rag_system" not in st.session_state:
        collector = DataBaseCollector(
            # host="localhost",
            # port=27017,
            database="confluence",
            collection_name="pages",
            context_size=200
        )
        
        # Initialize RAG system
        st.session_state.rag_system = RAGSystem(
            faiss_index_path="faiss.index",
            # mongo_host="localhost",
            # mongo_port=27017,
            mongo_db="confluence",
            mongo_collection="pages"
        )
        logging.info("RAG system initialized.")

    return st.session_state.rag_system

def handle_user_input(rag_system, user_input: str):
    """Process user input and generate response."""
    if user_input:
        try:
            # Show "thinking" indicator
            with st.spinner("Thinking..."):
                # Generate response
                start = time.time()
                response = rag_system.generate_response(user_input)
                logging.info(f"Response generated in {time.time() - start:.2f} seconds.")
            
            # Update conversation memory
            start = time.time()
            st.session_state.conversation_memory.append((user_input, response))
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})
            logging.info(f"Conversation memory updated in {time.time() - start:.2f} seconds.")
            
            # Display response with fading-in animation
            response_placeholder = st.empty()
            response_placeholder.markdown(f"<div style='opacity: 0; transition: opacity 1s;'>{response}</div>", unsafe_allow_html=True)
            time.sleep(0.1)  # Small delay to allow the DOM to update
            response_placeholder.markdown(f"<div style='opacity: 1; transition: opacity 1s;'>{response}</div>", unsafe_allow_html=True)
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            st.error("An error occurred while generating the response.")

def display_chat_history(user_emoticon: str):
    """Display chat messages using Streamlit's chat message containers."""
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user", avatar=user_emoticon).markdown(message["content"])
        else:
            st.chat_message("assistant", avatar="🐐").markdown(message["content"])

def main():
    # Initialize session state and RAG system
    initialize_session_state()
    rag_system = initialize_rag_system()
    st.title("RAG System Chatbot Demo")

    # Add sidebar with hyperlinks
    st.sidebar.title("Sources of Truth")
    # make me some text
    st.sidebar.markdown("Visit the sources of truth for more information about where the RAG system gets its knowledge. You could use this to check if the model is correct and to ask questions about the content.")
    st.sidebar.markdown("[The Secret Society of Rainbow Lemurs](https://tedsteketee.atlassian.net/wiki/spaces/~712020ec917477c3b543c198b7c9c1bd03fd16/pages/163978/The+Secret+Society+of+Rainbow+Lemurs)")
    st.sidebar.markdown("[The Enigmatic History of the Lost City of Quixalot](https://tedsteketee.atlassian.net/wiki/spaces/~712020ec917477c3b543c198b7c9c1bd03fd16/pages/131097/The+Enigmatic+History+of+the+Lost+City+of+Quixalot)")

    # make an option to choose the emoticon for the user
    st.sidebar.title("User Emoticon")
    st.sidebar.markdown("Choose your user emoticon!")
    user_emoticon = st.sidebar.selectbox("Select your user emoticon", ["🧑", "👩", "👨", "👩‍🦰", "👨‍🦰", "👩‍🦱", "👨‍🦱", "👩‍🦲", "👨‍🦲", "👩‍🦳", "👨‍🦳", "👱‍♀️", "👱‍♂️", "🧔", "👵", "👴", "👶", "👧", "🧒", "👦", "👩‍🦰", "👨‍🦰", "👩‍🦱", "👨‍🦱", "👩‍🦲", "👨‍🦲", "👩‍🦳", "👨‍🦳", "👱‍♀️", "👱‍♂️", "🧔", "👵", "👴", "👶", "👧", "🧒", "👦"])
    st.session_state.user_emoticon = user_emoticon


    # Display chat history
    display_chat_history(user_emoticon)

    # Chat input
    if user_input := st.chat_input("Type your question here..."):
        handle_user_input(rag_system, user_input)
        st.rerun()

if __name__ == "__main__":
    main()