import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import tensorflow as tf
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from huggingface_hub import InferenceClient
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from transformers import pipeline

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

tf.compat.v1.reset_default_graph()


def load_local_model_and_tokenizer(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    Load a local model and tokenizer for text generation.
    Replace `model_name` with the model you have downloaded locally.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically map the model to available GPUs
        torch_dtype="auto"  # Use the appropriate data type (e.g., fp16 if supported)
    )

    # Create a text generation pipeline
    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # max_length=2048,  # Adjust max token length
        temperature=0.3,  # Control randomness
        top_k=50,  # Sampling strategy
    )

    return generation_pipeline

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-nli-stsb-mean-tokens")    
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    
    client = InferenceClient(token=HUGGINGFACEHUB_API_TOKEN, 
                             model="meta-llama/Meta-Llama-3-8B-Instruct",
                             )
    """
    Set up the conversational chain using the locally loaded model.
    """
    # Load the local model and tokenizer
    # generation_pipeline = load_local_model_and_tokenizer()

    # # Wrap the pipeline in a HuggingFacePipeline for LangChain compatibility
    # llm = HuggingFacePipeline(pipeline=generation_pipeline)


    # llm = HuggingFaceEndpoint(
    #     task="text-generation",
    #     # repo_id="meta-llama/Llama-3.1-8B-Instruct",
    #     repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",
    #     max_new_tokens=4096,
    #     temperature= 0.1,
    #     repetition_penalty=1.05,
    #     huggingfacehub_api_token= HUGGINGFACEHUB_API_TOKEN
    # )

    # model = ChatHuggingFace(llm = llm)

    llm = HuggingFaceHub(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={"temperature": 0.3, "max_length": 2048},
        client=client,
        task = "text-generation"
    )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



def handle_userinput(user_question):
    print(f"----user_question:{user_question}-----")
    response = st.session_state.conversation.invoke({'question': user_question})
    print(response.keys)
    st.session_state.chat_history = response['chat_history']

    # # Extract and print only the response, not the context or chat history
    # answer = response.get('answer', "Sorry, I don't have an answer for that.")
    
    # # Display the answer in the chat
    # with st.chat_message("assistant"):
    #     st.write(answer)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)
            

def main():

    """Main Streamlit app."""

    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
        st.session_state.chat_history = []

    # App header
    st.header("Chat with multiple PDFs :brain:")

    # User question input
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                # Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs) if pdf_docs else "No documents uploaded."
                text_chunks = get_text_chunks(raw_text)

                # Create vector store and conversation chain
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

            st.success("Processing complete!")

    # Ensure default conversation chain if no PDFs are uploaded
    if st.session_state.conversation is None:
        default_text = "This is a default text to chat with the assistant."
        vectorstore = get_vectorstore(get_text_chunks(default_text))
        st.session_state.conversation = get_conversation_chain(vectorstore)


    # # Initialize session state if not present
    # if 'conversation' not in st.session_state or st.session_state.conversation is None:
    #     st.session_state.conversation = None
    #     st.session_state.chat_history = []

    # # Header
    # st.header("Chat with multiple PDFs :hammer_and_wrench:")

    # user_question = st.chat_input("Ask a question about your documents:")
    # if user_question:
    #     handle_userinput(user_question)

    # # Sidebar for file upload
    # with st.sidebar:
    #     st.subheader("Your documents")
    #     pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

    #     if not pdf_docs and st.session_state.conversation is None:
    #             default_text = "This is a default text to chat with the assistant."
    #             text_chunks = get_text_chunks(default_text)
    #             vectorstore = get_vectorstore(text_chunks)
    #             st.session_state.conversation = get_conversation_chain(vectorstore)
    #     else:

    #         if st.button("Process"):            
    #             with st.spinner("Processing..."):
                    
    #                 # Get PDF text
    #                 raw_text = get_pdf_text(pdf_docs)

    #                 # Get the text chunks
    #                 text_chunks = get_text_chunks(raw_text) if pdf_docs else ["This is a default text to chat with the assistant."]

    #                 # Create vector store
    #                 vectorstore = get_vectorstore(text_chunks)

    #                 # Create conversation chain
    #                 st.session_state.conversation = get_conversation_chain(vectorstore)
                
    #                 # Check if the conversation chain already exists, if not, create it
    #             if 'conversation' not in st.session_state or st.session_state.conversation is None:
    #                 st.session_state.conversation = get_conversation_chain(vectorstore)



if __name__ == '__main__':
    main()

