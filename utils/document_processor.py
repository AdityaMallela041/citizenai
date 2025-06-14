import streamlit as st
import sqlite3
import uuid
import datetime
import hashlib
import base64
from io import BytesIO
import mammoth
import PyPDF2
import docx
import pandas as pd
from transformers import pipeline

class DocumentProcessor:
    def __init__(self, db_connection, model, tokenizer):
        self.db_connection = db_connection
        self.model = model
        self.tokenizer = tokenizer
        self.init_database()
        
    def init_database(self):
        """Initialize document processing database tables"""
        c = self.db_connection.cursor()
        
        # Create documents table
        c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            content TEXT,
            summary TEXT,
            key_points TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed BOOLEAN DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')
        
        # Create document queries table
        c.execute('''
        CREATE TABLE IF NOT EXISTS document_queries (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            query TEXT NOT NULL,
            response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        ''')
        
        self.db_connection.commit()
    
    def extract_text_from_file(self, uploaded_file):
        """Extract text from uploaded file based on file type"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                return self._extract_from_pdf(uploaded_file)
            elif file_extension == 'docx':
                return self._extract_from_docx(uploaded_file)
            elif file_extension == 'txt':
                return str(uploaded_file.read(), "utf-8")
            else:
                return None
                
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            return None
    
    def _extract_from_pdf(self, uploaded_file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except:
            return None
    
    def _extract_from_docx(self, uploaded_file):
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except:
            return None
    
    def process_document(self, user_id, uploaded_file):
        """Process uploaded document and generate summary"""
        try:
            # Extract text
            content = self.extract_text_from_file(uploaded_file)
            if not content:
                return None, "Could not extract text from document"
            
            # Generate summary and key points using the model
            summary = self._generate_summary(content)
            key_points = self._extract_key_points(content)
            
            # Store in database
            doc_id = str(uuid.uuid4())
            c = self.db_connection.cursor()
            
            c.execute('''
                INSERT INTO documents (id, user_id, filename, file_type, content, summary, key_points, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (doc_id, user_id, uploaded_file.name, uploaded_file.type, 
                  content, summary, key_points, 1))
            
            self.db_connection.commit()
            return doc_id, "Document processed successfully"
            
        except Exception as e:
            return None, f"Error processing document: {str(e)}"
    
    def _generate_summary(self, content):
        """Generate AI summary of document content"""
        try:
            max_length = 1000
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            prompt = f"Summarize this government document in 2-3 sentences:\n\n{content}\n\nSummary:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

            # ✅ Ensure inputs are on same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with st.spinner("Generating summary..."):
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Summary:" in summary:
                summary = summary.split("Summary:")[-1].strip()
            
            return summary[:500]
        
        except Exception as e:
            return f"Summary generation failed: {str(e)}"

    
    def _extract_key_points(self, content):
        """Extract key points from document"""
        try:
            # Simple keyword extraction based on common government document patterns
            key_phrases = []
            
            # Look for common patterns in government documents
            patterns = [
                "requirements", "procedures", "deadlines", "applications",
                "eligibility", "benefits", "regulations", "compliance",
                "fees", "documents needed", "contact information"
            ]
            
            lines = content.lower().split('\n')
            for line in lines:
                for pattern in patterns:
                    if pattern in line and len(line) < 200:
                        key_phrases.append(line.strip())
                        break
            
            return "; ".join(key_phrases[:5])  # Limit to 5 key points
            
        except Exception as e:
            return f"Key point extraction failed: {str(e)}"
    
    def get_user_documents(self, user_id):
        """Get all documents for a user"""
        c = self.db_connection.cursor()
        c.execute('''
            SELECT id, filename, file_type, summary, upload_date, processed
            FROM documents
            WHERE user_id = ?
            ORDER BY upload_date DESC
        ''', (user_id,))
        
        return c.fetchall()
    
    def get_document_details(self, doc_id, user_id):
        """Get detailed information about a specific document"""
        c = self.db_connection.cursor()
        c.execute('''
            SELECT * FROM documents
            WHERE id = ? AND user_id = ?
        ''', (doc_id, user_id))
        
        return c.fetchone()
    
    def query_document(self, doc_id, user_id, query):
        """Answer questions about a specific document"""
        try:
            # Get document content
            document = self.get_document_details(doc_id, user_id)
            if not document:
                return "Document not found"
            
            content = document[4]  # content column
            
            # Generate response using the model
            prompt = f"Based on this document content, answer the question:\n\nDocument: {content[:800]}...\n\nQuestion: {query}\n\nAnswer:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            with st.spinner("Analyzing document..."):
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + 150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
            # Store query and response
            query_id = str(uuid.uuid4())
            c = self.db_connection.cursor()
            c.execute('''
                INSERT INTO document_queries (id, document_id, user_id, query, response)
                VALUES (?, ?, ?, ?, ?)
            ''', (query_id, doc_id, user_id, query, response))
            
            self.db_connection.commit()
            
            return response[:500]  # Limit response length
            
        except Exception as e:
            return f"Error querying document: {str(e)}"
    
    def get_document_queries(self, doc_id, user_id):
        """Get all queries for a specific document"""
        c = self.db_connection.cursor()
        c.execute('''
            SELECT query, response, created_at
            FROM document_queries
            WHERE document_id = ? AND user_id = ?
            ORDER BY created_at DESC
        ''', (doc_id, user_id))
        
        return c.fetchall()

def render_document_processor(document_processor, user_id):
    """Render the document processing interface"""
    st.title("📄 Smart Document Processing")
    st.write("Upload government documents, forms, or policies to get AI-powered analysis and answers to your questions.")
    
    # Create tabs for different functionalities
    upload_tab, my_docs_tab = st.tabs(["📤 Upload Document", "📂 My Documents"])
    
    with upload_tab:
        st.subheader("Upload New Document")
        
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, Word documents, and text files"
        )
        
        if uploaded_file is not None:
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            
            # Show file details
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**File Name:** {uploaded_file.name}")
                st.info(f"**File Type:** {uploaded_file.type}")
            with col2:
                st.info(f"**File Size:** {uploaded_file.size / 1024:.1f} KB")
            
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document... This may take a moment."):
                    doc_id, message = document_processor.process_document(user_id, uploaded_file)
                    
                    if doc_id:
                        st.success(message)
                        st.session_state.processed_doc_id = doc_id
                        st.rerun()
                    else:
                        st.error(message)
    
    with my_docs_tab:
        st.subheader("Your Processed Documents")
        
        documents = document_processor.get_user_documents(user_id)
        
        if documents:
            for doc in documents:
                with st.expander(f"📄 {doc[1]} - {doc[4]}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Summary:** {doc[3]}")
                        st.write(f"**Uploaded:** {doc[4]}")
                        st.write(f"**Status:** {'✅ Processed' if doc[5] else '⏳ Processing'}")
                    
                    with col2:
                        if st.button(f"View Details", key=f"view_{doc[0]}"):
                            st.session_state.selected_doc_id = doc[0]
                    
                    # Document query section
                    if doc[5]:  # If processed
                        st.markdown("---")
                        st.write("**Ask questions about this document:**")
                        
                        query = st.text_input(
                            "Your question:",
                            placeholder="e.g., What are the key requirements?",
                            key=f"query_{doc[0]}"
                        )
                        
                        if st.button("Ask", key=f"ask_{doc[0]}"):
                            if query:
                                response = document_processor.query_document(doc[0], user_id, query)
                                st.write(f"**Answer:** {response}")
                            else:
                                st.warning("Please enter a question")
                        
                        # Show previous queries
                        queries = document_processor.get_document_queries(doc[0], user_id)
                        if queries:
                            st.write("**Previous Questions:**")
                            for q in queries[:3]:  # Show last 3 queries
                                st.write(f"❓ {q[0]}")
                                st.write(f"💬 {q[1]}")
                                st.caption(f"Asked on: {q[2]}")
                                st.write("---")
        else:
            st.info("No documents uploaded yet. Use the 'Upload Document' tab to get started!")
    
    # Show processing result if available
    if hasattr(st.session_state, 'processed_doc_id'):
        doc_details = document_processor.get_document_details(st.session_state.processed_doc_id, user_id)
        if doc_details:
            st.success("Document processed successfully!")
            st.write(f"**Summary:** {doc_details[5]}")
            st.write(f"**Key Points:** {doc_details[6]}")
            del st.session_state.processed_doc_id