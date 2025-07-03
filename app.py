import monkey_patch_sqlite  
import streamlit as st
import tempfile
import os
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = None

if 'embedding' not in st.session_state:
    st.session_state.embedding = None

if 'llm' not in st.session_state:
    st.session_state.llm = None

@st.cache_resource
def load_embedding():
    """
    Load the multilingual embedding model used for document chunking and similarity search.

    Returns:
        HuggingFaceEmbeddings: The embedding model.
    """
    return HuggingFaceEmbeddings(
        model='intfloat/multilingual-e5-base',
        model_kwargs={"device": "cpu"}  # Use CPU to avoid GPU OOM errors
    )

def load_llm():
    """
    Load and quantize the Vicuna-7B LLM model for inference using HuggingFace pipeline.

    Returns:
        HuggingFacePipeline: A LangChain-compatible LLM pipeline.
    """
    MODEL_NAME = 'lmsys/vicuna-7b-v1.5'

    config_nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=config_nf4,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model_pipeline = pipeline(
        'text-generation',
        model=model,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        tokenizer=tokenizer,
        device_map='auto'
    )

    llm = HuggingFacePipeline(
        pipeline=model_pipeline
    )

    return llm

def process_pdf(file_uploaded):
    """
    Handle PDF loading, semantic chunking, vectorization, and RAG chain construction.

    Args:
        file_uploaded (UploadedFile): The uploaded PDF file from Streamlit.

    Returns:
        tuple: (rag_chain, num_chunks) where:
            - rag_chain: LangChain RAG pipeline for question answering.
            - num_chunks: Number of document chunks created.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(file_uploaded.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    document = loader.load()

    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embedding,
        add_start_index=True,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500
    )

    docs = semantic_splitter.split_documents(document)
    vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embedding)
    retriever = vector_db.as_retriever()

    prompt = hub.pull('rlm/rag-prompt')

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)

    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )

    os.unlink(tmp_file_path)
    return rag_chain, len(docs)

def main():
    """
    The main entry point for the Streamlit web app.
    Handles UI rendering, model loading, PDF processing, and user interaction.
    """
    st.set_page_config(page_title="PDF RAG Assistance", layout='wide')
    st.title("PDF RAG Assistant")

    st.markdown("""
    **Ứng dụng AI giúp bạn hỏi đáp trực tiếp với nội dung tài liệu PDF bằng tiếng Việt**

    **Cách sử dụng đơn giản:**
    1. **Upload PDF** → Chọn file PDF từ máy tính và nhấn "Xử lý PDF"
    2. **Đặt câu hỏi** → Nhập câu hỏi về nội dung tài liệu và nhận câu trả lời ngay lập tức

    ---
    """)

    if not st.session_state.models_loaded:
        st.info("Loading models")
        st.session_state.embedding = load_embedding()
        st.session_state.llm = load_llm()
        st.session_state.models_loaded = True
        st.success("Model Preparation is Done!")
        st.rerun()

    uploaded_file = st.file_uploader("Upload your PDF file", type='pdf')
    if uploaded_file and st.button("Press to process file"):
        st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
        st.success(f"The semantic split on file is done. Number of chunks is {num_chunks}")

    question = st.text_input("Please input a question")
    if question:
        with st.spinner("Answering question..."):
            output = st.session_state.rag_chain.invoke(question)
            answer = output.split("Answer: ")[1] if "Answer: " in output else output
            st.write("The answer is:")
            st.write(answer)

if __name__ == '__main__':
    main()
