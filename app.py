import shutil
from dotenv import load_dotenv, find_dotenv
import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma


load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
CHROMA_PATH = "ChromaDB"

PROMPT_TEMPLATE = """
Answer the question based only on the context provided below:

{context}

---

Answer the following question based on the above context: {question}
"""

repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct'

# Initialize the model
client = InferenceClient(token=HUGGINGFACEHUB_API_TOKEN, model = repo_id)
llm = HuggingFaceEndpoint(
    task="text-generation",
    repo_id = repo_id,
    max_new_tokens=4096,
    temperature=0.5,
    repetition_penalty=1.05,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)


print(llm.invoke("what is the capital of France?"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    if not text.strip():
        raise ValueError("Uploaded PDFs contain no extractable text.")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=5000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_text(text)

def safe_rmtree(path, retries=5, delay=1):
    for i in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise

def Create_vectorstore(text_chunks):
    try:
            if os.path.exists(CHROMA_PATH):
                safe_rmtree(CHROMA_PATH)

            db = Chroma(
                collection_name="pdfs",
                embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                persist_directory=CHROMA_PATH,
            )

            db.add_texts(text_chunks)

            return db

    except Exception as e:
            st.error(f"Failed to initialize vectorstore: {e}")
            return None

def Handle_userinput(question):
    if not question.strip():
        return "Please ask a valid question."
    
    if not st.session_state.vectorstore:
        return "Vectorstore is not initialized. Please upload and process files first."

    context = st.session_state.vectorstore.similarity_search_with_relevance_scores(question)
    if not context:
        return "The information you requested is not available in the documents provided."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _scores in context])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
    print(f"\n\n-----{prompt}\n\n-----\n")
    response = llm.invoke(prompt)
    print(f"\n\n----{response}\n\n----\n")
    return response if response else "No response from the model."

def main():
    st.set_page_config(page_title="Chat with Llama", page_icon="🦙")
    
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs 🧠")

    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        with st.chat_message("User"):
            st.write(user_question)
        st.session_state.chat_history.append({"role" : "User", "content" : user_question})
        response = Handle_userinput(user_question)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("Assistant"):
            st.write(response)

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Delete Vectorstore") and st.session_state.vectorstore:
            st.session_state.vectorstore.delete_collection()
        if st.button("Process") and uploaded_files:
            try:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(uploaded_files)
                    st.info("text extraction complete!")
                    text_chunks = get_text_chunks(raw_text)
                    st.info("text chunks are created successfully!")
                    st.session_state.vectorstore = Create_vectorstore(text_chunks)
                    st.success("Processing complete! - Vectorstore is ready for use.")
            except Exception as e:
                st.error(f"Failed to process files: {e}")
        

if __name__ == "__main__":
    main()
