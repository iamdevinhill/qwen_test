#!/usr/bin/env python3
"""
Simple RAG application using Ollama Qwen3-VL and ChromaDB
Tests the context window of the model with PDF content
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
import ollama
from PyPDF2 import PdfReader


class RAGApp:
    def __init__(self, pdf_path: str, collection_name: str = "pdf_rag"):
        self.pdf_path = pdf_path
        self.collection_name = collection_name
        self.client = chromadb.Client()
        
        # Use default embedding function
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Initialize collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        
        self.model = "qwen3-vl:32b"
        
        # Create logs directory
        self.logs_dir = Path("query_logs")
        self.logs_dir.mkdir(exist_ok=True)
        
    def extract_text_from_pdf(self) -> str:
        """Extract all text from PDF"""
        print(f"📄 Loading PDF: {self.pdf_path}")
        reader = PdfReader(self.pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"\n--- Page {i+1} ---\n"
            text += page.extract_text()
        print(f"✅ Extracted {len(text)} characters from {len(reader.pages)} pages")
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def index_pdf(self):
        """Load PDF, chunk it, and store in ChromaDB"""
        # Check if already indexed
        if self.collection.count() > 0:
            print(f"⚠️  Collection already has {self.collection.count()} documents")
            response = input("Re-index? (y/n): ").lower()
            if response != 'y':
                return
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        
        # Extract and chunk text
        text = self.extract_text_from_pdf()
        chunks = self.chunk_text(text)
        
        # Store in ChromaDB
        print("💾 Indexing chunks in ChromaDB...")
        self.collection.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"chunk_id": i, "source": self.pdf_path} for i in range(len(chunks))]
        )
        print(f"✅ Indexed {len(chunks)} chunks successfully!")
    
    def query(self, question: str, n_results: int = 5) -> str:
        """Query the RAG system"""
        # Retrieve relevant chunks
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        # Build context from retrieved chunks
        context_chunks = results['documents'][0]
        context = "\n\n".join(context_chunks)
        
        context_length = len(context)
        print(f"\n🔍 Retrieved {len(context_chunks)} relevant chunks ({context_length} chars)")
        
        # Build prompt with context
        prompt = f"""Based on the following context from the document, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt_length = len(prompt)
        print(f"📊 Total prompt length: {prompt_length} characters")
        print("🤖 Querying Ollama Qwen3-VL...")
        
        # Query Ollama
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'num_predict': -1,  # No limit on tokens
                }
            )
            
            answer = response['message']['content']
            
            # Extract token information
            token_info = {
                'prompt_eval_count': response.get('prompt_eval_count', 0),
                'eval_count': response.get('eval_count', 0),
                'total_duration': response.get('total_duration', 0),
                'load_duration': response.get('load_duration', 0),
                'prompt_eval_duration': response.get('prompt_eval_duration', 0),
                'eval_duration': response.get('eval_duration', 0),
            }
            
            print(f"🔢 Tokens - Prompt: {token_info['prompt_eval_count']}, Response: {token_info['eval_count']}")
            
            # Save the query log
            self.save_query_log(question, answer, context_length, prompt_length, token_info)
            
            return answer
        except Exception as e:
            error_msg = f"Error querying model: {str(e)}"
            self.save_query_log(question, error_msg, context_length, prompt_length, None)
            return error_msg
    
    def save_query_log(self, question: str, answer: str, context_length: int, prompt_length: int, token_info: dict = None):
        """Save query, response and metadata to timestamped JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "question": question,
            "response": answer,
            "metadata": {
                "context_length_chars": context_length,
                "prompt_length_chars": prompt_length,
                "pdf_source": self.pdf_path
            }
        }
        
        # Add token details if available
        if token_info:
            log_data["token_details"] = {
                "prompt_tokens": token_info['prompt_eval_count'],
                "response_tokens": token_info['eval_count'],
                "total_tokens": token_info['prompt_eval_count'] + token_info['eval_count'],
                "timing": {
                    "total_duration_ns": token_info['total_duration'],
                    "load_duration_ns": token_info['load_duration'],
                    "prompt_eval_duration_ns": token_info['prompt_eval_duration'],
                    "eval_duration_ns": token_info['eval_duration'],
                    "total_duration_sec": token_info['total_duration'] / 1e9,
                    "eval_duration_sec": token_info['eval_duration'] / 1e9,
                }
            }
        
        log_file = self.logs_dir / f"query_{timestamp}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Saved to: {log_file}")
    
    def run_interactive(self):
        """Run interactive terminal session"""
        print("\n" + "="*60)
        print("🚀 RAG App - Testing Qwen3-VL Context Window")
        print("="*60)
        print(f"📚 PDF: {self.pdf_path}")
        print(f"🗄️  Vector Store: ChromaDB ({self.collection.count()} chunks)")
        print(f"🤖 Model: {self.model}")
        print("\nCommands:")
        print("  - Type your question to query the PDF")
        print("  - 'stats' - Show collection statistics")
        print("  - 'quit' or 'exit' - Exit the app")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("❓ Question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if question.lower() == 'stats':
                    count = self.collection.count()
                    print(f"\n📊 Collection Statistics:")
                    print(f"  - Total chunks: {count}")
                    print(f"  - Collection name: {self.collection_name}")
                    continue
                
                # Process query
                answer = self.query(question)
                print(f"\n💡 Answer:\n{answer}\n")
                print("-"*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")


def main():
    # Find PDF in current directory
    pdf_files = list(Path('.').glob('*.pdf'))
    
    if not pdf_files:
        print("❌ No PDF files found in current directory!")
        sys.exit(1)
    
    # Use first PDF found
    pdf_path = str(pdf_files[0])
    
    # Initialize RAG app
    app = RAGApp(pdf_path)
    
    # Index the PDF
    app.index_pdf()
    
    # Run interactive session
    app.run_interactive()


if __name__ == "__main__":
    main()
