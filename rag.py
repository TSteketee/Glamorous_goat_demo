import os
import numpy as np
import requests
from dotenv import load_dotenv

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from transformers import BertTokenizer, BertModel
import torch
import faiss

from groq import Groq
from bs4 import BeautifulSoup
import re

import logging


class TextVectorizer:
    def __init__(self, model_name):
        self.model, self.tokenizer = self._load_model(model_name)
        self.model.eval()

    def _load_model(self, model_name):
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            if "transformers" in logger.name.lower():
                logger.setLevel(logging.ERROR)
        model = BertModel.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        return model, tokenizer
    
    def vectorize(self, text):
        # Handle empty or invalid text
        if not text or not isinstance(text, str):
            return np.zeros((1, 768))  # BERT base dimension
            
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    def vectorize_batch(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        elif not texts:
            return np.zeros((1, 768))
            
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()


class DataBaseCollector:
    def __init__(self, database, collection_name, context_size):
        load_dotenv()
        self.api_key = os.getenv("CONFLUENCE_TOKEN")
        self.api_email = os.getenv("CONFLUENCE_EMAIL")
        self.api_base_url = os.getenv("CONFLUENCE_BASE_URL")
        self.uri = uri = os.getenv("MONGO_URI")

        # Validate environment variables
        if not all([self.api_key, self.api_email, self.api_base_url]):
            raise ValueError("Missing required environment variables")
        

        # Connect to MongoDB Atlas
        self.client = MongoClient(self.uri, server_api=ServerApi('1'))
        self.db = self.client[database]
        self.collection = self.db[collection_name]
        
        # Create indexes for better performance
        self.collection.create_index("page_id")
        
        self.vectorizer = TextVectorizer("bert-base-uncased")
        self.context_size = context_size
        
    def _clean_html(self, html_content):
        """Clean HTML content and extract text."""
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ')
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def _get_confluence_page(self, page_id):
        url = f"{self.api_base_url}content/{page_id}?expand=body.storage"
        auth = requests.auth.HTTPBasicAuth(self.api_email, self.api_key)
        headers = {'Accept': 'application/json'}
        
        try:
            response = requests.get(url, headers=headers, auth=auth)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching page {page_id}: {str(e)}")
            raise

    def _create_chunks(self, text):
        """Create overlapping chunks of text."""
        words = text.split()
        chunks = []
        chunk_indices = []
        
        for i in range(0, len(words), self.context_size // 2):  # 50% overlap
            chunk = ' '.join(words[i:i + self.context_size])
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
                chunk_indices.append(i)
                
        return chunks, chunk_indices

    def update(self, page_id):
        try:
            if isinstance(page_id, list):
                for pid in page_id:
                    self._update_single_page(pid)
            else:
                self._update_single_page(page_id)
                
            # Update FAISS index after all pages are processed
            self._update_faiss_index()
            
        except Exception as e:
            logging.error(f"Error in update process: {str(e)}")
            raise

    def _update_single_page(self, page_id):
        """Update a single page in the database."""
        page_data = self._get_confluence_page(page_id)
        title = page_data['title']
        content = self._clean_html(page_data['body']['storage']['value'])
        
        # Create chunks with overlap
        full_text = f"{title}. {content}"
        chunks, chunk_indices = self._create_chunks(full_text)
        
        # Vectorize chunks
        if chunks:
            vectors = self.vectorizer.vectorize_batch(chunks)
            
            # Store in MongoDB
            doc = {
                "page_id": page_id,
                "title": title,
                "content": content,
                "chunks": chunks,
                "chunk_indices": chunk_indices,
                "vectors": vectors.tolist()
            }
            
            self.collection.update_one(
                {"page_id": page_id},
                {"$set": doc},
                upsert=True
            )

    def _update_faiss_index(self):
        """Update the FAISS index with all vectors."""
        try:
            # Get dimension from a sample vector
            sample_doc = self.collection.find_one({"vectors": {"$exists": True}})
            if not sample_doc or not sample_doc.get("vectors"):
                raise ValueError("No vectors found in database")
                
            vector_dim = len(sample_doc["vectors"][0])
            index = faiss.IndexFlatL2(vector_dim)
            
            # Collect all vectors and their mappings
            all_vectors = []
            vector_mappings = []
            
            for doc in self.collection.find({"vectors": {"$exists": True}}):
                if doc.get("vectors"):
                    vectors = np.array(doc["vectors"])
                    all_vectors.append(vectors)
                    
                    # Create mapping for each vector
                    for i in range(len(vectors)):
                        vector_mappings.append({
                            "page_id": doc["page_id"],
                            "chunk_index": i,
                            "title": doc["title"]
                        })
            
            if all_vectors:
                all_vectors = np.vstack(all_vectors)
                index.add(all_vectors)
                
                # Save index and mappings
                faiss.write_index(index, "faiss.index")
                self.collection.update_one(
                    {"_id": "vector_mappings"},
                    {"$set": {"mappings": vector_mappings}},
                    upsert=True
                )
                
        except Exception as e:
            logging.error(f"Error updating FAISS index: {str(e)}")
            raise


class RAGSystem:
    def __init__(self, faiss_index_path, mongo_db, mongo_collection):
        # Initialize components
        try:
            load_dotenv()
            self.uri = os.getenv("MONGO_URI")
            self.index = faiss.read_index(faiss_index_path)
            self.client = MongoClient(self.uri, server_api=ServerApi('1'))
            self.db = self.client[mongo_db]
            self.collection = self.db[mongo_collection]
            
            # Load vector mappings
            mapping_doc = self.collection.find_one({"_id": "vector_mappings"})
            if not mapping_doc:
                raise ValueError("Vector mappings not found in database")
            self.vector_mappings = mapping_doc["mappings"]
            
            self.vectorizer = TextVectorizer("bert-base-uncased")
            
            load_dotenv()
            groq_key = os.getenv("GROQ_KEY")
            if not groq_key:
                raise ValueError("GROQ_KEY not found in environment variables")
            self.groq_client = Groq(api_key=groq_key)
            
        except Exception as e:
            logging.error(f"Error initializing RAG system: {str(e)}")
            raise

    def retrieve_relevant_chunks(self, query, top_k=5):
        """Retrieve the most relevant chunks for a query."""
        try:
            # Vectorize query
            query_vector = self.vectorizer.vectorize(query)
            
            # Search in FAISS
            D, I = self.index.search(query_vector, top_k)
            
            relevant_chunks = []
            for idx in I[0]:
                if idx >= 0 and idx < len(self.vector_mappings):
                    mapping = self.vector_mappings[idx]
                    doc = self.collection.find_one({"page_id": mapping["page_id"]})
                    if doc and mapping["chunk_index"] < len(doc["chunks"]):
                        chunk = doc["chunks"][mapping["chunk_index"]]
                        relevant_chunks.append({
                            "text": chunk,
                            "title": doc["title"],
                            "score": float(D[0][len(relevant_chunks)])
                        })
            
            return relevant_chunks
            
        except Exception as e:
            logging.error(f"Error retrieving chunks: {str(e)}")
            return []

    def generate_response(self, query, temperature=0.2):
        """Generate a response using retrieved context and Groq."""
        try:
            relevant_chunks = self.retrieve_relevant_chunks(query)
            
            if not relevant_chunks:
                return "I couldn't find any relevant information to answer your question."
            
            # Format context with source information
            context = "\n\n".join([
                f"From '{chunk['title']}':\n{chunk['text']}"
                for chunk in relevant_chunks
            ])
            
            prompt = f"""Please answer the following question based on the provided context. If the context doesn't contain enough relevant information, please say so.

Context:
{context}

Question: {query}

Please provide a detailed answer, citing specific information from the context where relevant."""

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate answers based on the given context. Always acknowledge the source of information and maintain accuracy."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama3-8b-8192",
                temperature=temperature  # Lower temperature for more focused responses
            )

            return chat_completion.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "I encountered an error while trying to generate a response. Please try again."
        

if __name__ == "__main__":
    collector = DataBaseCollector(
        # host="localhost",
        # port=27017,
        database="confluence",
        collection_name="pages",
        context_size=200
    )
    
    collector.update([163978, 131097])
