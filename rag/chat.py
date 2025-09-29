import streamlit as st
import requests
import json
import base64
from io import BytesIO

# Import LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
# API Key must be defined as an empty string to use the environment's provided token
API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_FLASH = "gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Utility Functions for Gemini API Calls ---

def call_gemini_api(model_name, contents, system_prompt=None):
    """
    Generic function to call the Gemini API.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
    
    # Structure contents for the API
    parts = contents
    
    # Base payload
    payload = {
        "contents": [{"parts": parts}],
    }

    # Add system instruction if provided
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    try:
        # Simple retry mechanism for robustness
        for attempt in range(3):
            response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Error: No text response from model.')
                return text
            
            elif response.status_code == 429:
                st.warning(f"Rate limit hit. Retrying in {2**attempt} seconds...")
                st.session_state.gemini_status = f"Rate limit hit. Retrying in {2**attempt} seconds..."
                import time
                time.sleep(2**attempt)
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                st.session_state.gemini_status = f"API Error ({response.status_code}): {response.text}"
                return None

        st.error("Failed to get response from Gemini API after multiple retries.")
        st.session_state.gemini_status = "Failed after retries."
        return None

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        st.session_state.gemini_status = f"Request failed: {e}"
        return None


def get_image_description(image_bytes, mime_type, filename):
    """
    Uses the Gemini API to describe an image and perform OCR/captioning.
    """
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # Contents for multimodal input
    contents = [
        {"text": f"Analyze this image named '{filename}'. Describe its content, extract any text (OCR), and summarize the key information. Use this summary as the document content."},
        {"inlineData": {"data": base64_image, "mimeType": mime_type}}
    ]
    
    st.session_state.gemini_status = f"Generating description for {filename}..."
    description = call_gemini_api(GEMINI_FLASH, contents)
    st.session_state.gemini_status = "Description generated."
    return description

# --- Document Processing Functions ---

def process_uploaded_files(uploaded_files):
    """
    Processes uploaded files (PDFs, Images) into text chunks.
    """
    all_chunks = []
    
    # 1. Initialize Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 2. Process Files
    for file in uploaded_files:
        filename = file.name
        st.session_state.gemini_status = f"Processing file: {filename} ({file.type})..."

        try:
            if file.type == "application/pdf":
                # Save uploaded file to disk temporarily for PyPDFLoader
                with open(filename, "wb") as f:
                    f.write(file.getbuffer())
                
                # Load PDF
                loader = PyPDFLoader(file_path=filename)
                docs = loader.load()
                
                # Split documents and tag with metadata
                chunks = text_splitter.split_documents(documents=docs)
                for i, chunk in enumerate(chunks):
                    chunk.metadata['source'] = filename
                    chunk.metadata['chunk_index'] = i
                    chunk.metadata['page_label'] = chunk.metadata.get('page', 'N/A')
                
                all_chunks.extend(chunks)
                st.session_state.gemini_status = f"PDF '{filename}' processed. {len(chunks)} chunks created."

            elif file.type in ["image/jpeg", "image/png", "image/webp"]:
                image_bytes = file.getbuffer()
                
                # Get multimodal description
                description = get_image_description(image_bytes, file.type, filename)
                
                if description:
                    # Create a single document from the description
                    image_doc = Document(
                        page_content=description,
                        metadata={
                            'source': filename,
                            'chunk_index': 0,
                            'page_label': 'Description'
                        }
                    )
                    all_chunks.append(image_doc)
                    st.session_state.gemini_status = f"Image '{filename}' described and added."

            elif file.type.startswith("audio/"):
                st.warning(f"Audio file '{filename}' uploaded but skipping processing. RAG for audio requires a dedicated Speech-to-Text API, which is outside the scope of this simple RAG implementation.")

            else:
                st.warning(f"Skipping unsupported file type: {file.type} for {filename}.")
                
        except Exception as e:
            st.error(f"Error processing {filename}: {e}")
            st.session_state.gemini_status = f"Error processing {filename}: {e}"

    
    # 3. Create Vector Store
    if all_chunks:
        with st.spinner("Creating and indexing vector store (this might take a minute)..."):
            # Load HuggingFace Embeddings model
            embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME
            )
            
            # Create FAISS vector store in memory
            vector_store = FAISS.from_documents(
                documents=all_chunks,
                embedding=embedding_model
            )
            
            st.session_state.vector_db = vector_store
            st.session_state.documents_loaded = True
            st.session_state.gemini_status = f"Indexing complete. {len(all_chunks)} chunks indexed."
            st.success(f"Successfully indexed {len(all_chunks)} chunks from uploaded files!")
    else:
        st.error("No valid documents were processed.")
        st.session_state.gemini_status = "No valid documents were processed."


# --- Streamlit UI Setup ---

st.set_page_config(layout="wide", page_title="Multimodal RAG Chatbot")
st.title("📚 Multimodal RAG Chatbot")

# Initialize session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "gemini_status" not in st.session_state:
    st.session_state.gemini_status = "Ready to upload files."


# --- Sidebar for File Upload and Processing ---
with st.sidebar:
    st.header("1. Upload Files")
    
    uploaded_files = st.file_uploader(
        "Upload your PDFs, JPGs, or PNGs (max 5 files)",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    st.caption("Image files will be described by Gemini to extract content.")
    st.caption("Supported file types: PDF, JPG, PNG.")
    
    if st.button("Create RAG Index"):
        if uploaded_files:
            process_uploaded_files(uploaded_files)
        else:
            st.warning("Please upload files first.")

    st.markdown("---")
    st.header("System Status")
    st.info(st.session_state.gemini_status)
    if st.session_state.documents_loaded:
        st.success("Vector Store is ready! Ask your questions below.")
    else:
        st.warning("Upload files and click 'Create RAG Index' to begin chatting.")


# --- Main Chat Interface ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and RAG logic
if prompt := st.chat_input("Ask a question about the uploaded documents..."):
    # Check if vector store is ready
    if not st.session_state.documents_loaded or st.session_state.vector_db is None:
        st.error("Please process your files first using the sidebar.")
        # Do not save the prompt to history if we can't process it
        prompt = None 

    if prompt:
        # 1. Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Searching documents and generating response...")

            # 2. Retrieval (RAG)
            try:
                st.session_state.gemini_status = "Searching documents..."
                # Use k=4 for better context coverage
                search_results = st.session_state.vector_db.similarity_search(query=prompt, k=4)
                
                context_list = []
                for result in search_results:
                    context_list.append(
                        f"Page Content: {result.page_content.strip()} \n"
                        f"Source: {result.metadata.get('source', 'Unknown File')} \n"
                        f"Location: Page {result.metadata.get('page_label', 'N/A')}, Chunk {result.metadata.get('chunk_index', 'N/A')}"
                    )

                context = "\n\n---\n\n".join(context_list)
                
                # 3. Augmentation (Create System Prompt)
                SYSTEM_PROMPT = f"""
                You are a helpful AI assistant. Answer the user's question based ONLY on the provided context, which is retrieved from uploaded files.
                
                You MUST cite the source information (Source, Location) for every part of your answer. If the context does not contain the answer, state clearly that the information is not available in the documents.
                
                Context:
                {context}
                """

                # 4. Generation
                st.session_state.gemini_status = "Calling Gemini API for generation..."
                
                # Use the helper function to call Gemini
                gemini_response = call_gemini_api(
                    model_name=GEMINI_FLASH, 
                    contents=[{"text": prompt}],
                    system_prompt=SYSTEM_PROMPT
                )
                
                if gemini_response:
                    full_response = gemini_response
                    
                    # 5. Display response
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.session_state.gemini_status = "Response delivered."
                else:
                    message_placeholder.error("Could not generate a response from the model.")
                    st.session_state.gemini_status = "Model response failed."

            except Exception as e:
                error_msg = f"An error occurred during RAG process: {e}"
                message_placeholder.error(error_msg)
                st.session_state.gemini_status = error_msg

# --- Instructions and Disclaimer ---
st.markdown("---")
st.markdown(
    """
    ### How It Works:
    1. **Upload:** Upload one or more PDFs, JPGs, or PNGs.
    2. **Index:** Click 'Create RAG Index'. The app converts the files into text:
       - PDFs are loaded and split into text chunks.
       - Images are sent to the **Gemini Multimodal Model** to generate a descriptive text summary (including OCR).
    3. **Embed:** All text chunks are converted into vector embeddings using the `all-MiniLM-L6-v2` model and stored in an in-memory FAISS database.
    4. **Chat:** Ask a question. The app retrieves the most relevant chunks from the database and sends them to the **Gemini RAG Model** to generate a grounded answer.
    """
)
st.markdown(
    """
    *Note: This application runs all processing, including the vector store (FAISS), entirely in memory within the Streamlit session.*
    """
)
