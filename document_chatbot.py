import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="AI-Powered Document Chatbot", page_icon="📚", layout="wide")
st.title("📚 AI-Powered Document Chatbot")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OPENAI_API_KEY is missing. Please check your .env file!")
    st.stop()

if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file and not st.session_state.file_processed:
        with st.spinner("Processing document..."):
            with open("temp_doc.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            pdf_loader = PyPDFLoader("temp_doc.pdf")
            documents = pdf_loader.load()
            #documents = documents[3:8]
            if not documents:
                st.error("❌ No text extracted from the document! Check the PDF file.")
                st.stop()
                
            st.success(f"✅ Document loaded with {len(documents)} pages")
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
            text_chunks = text_splitter.split_documents(documents)
            
            if not text_chunks:
                st.error("❌ No text chunks available. Check text extraction.")
                st.stop()
                
            st.info(f"✅ Document split into {len(text_chunks)} chunks")
            
            with st.spinner("Creating embeddings..."):
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vector_store = FAISS.from_documents(text_chunks, embeddings)
                st.session_state.vector_store = vector_store
                
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                llm = ChatOpenAI(
                    model_name="gpt-4",
                    temperature=0.3,
                    max_tokens=1024,
                    openai_api_key=openai_api_key
                )
                
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=memory
                )
                
                st.session_state.file_processed = True
                st.success("✅ Document processed and ready for questions!")


if st.session_state.file_processed:
    tab1, tab2 = st.tabs(["Conversation", "Quick Insights"])
    
    with tab1:
        st.header("Ask questions about your document")
        user_question = st.text_input("Enter your question:")
        
        if user_question:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation.invoke({"question": user_question})
                st.write("Answer:")
                st.write(response["answer"])
    
    with tab2:
        st.header("Get Quick Insights")
        
        if st.button("Generate Document Summary"):
            with st.spinner("Generating summary..."):
                summary = st.session_state.conversation.invoke({"question": "Provide a concise summary of the document."})
                st.subheader("Summary")
                st.write(summary["answer"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Key Points"):
                with st.spinner("Analyzing..."):
                    key_points = st.session_state.conversation.invoke({"question": "What are the key points discussed in the document?"})
                    st.subheader("Key Points")
                    st.write(key_points["answer"])
        
        with col2:
            if st.button("Actionable Takeaways"):
                with st.spinner("Generating takeaways..."):
                    takeaways = st.session_state.conversation.invoke({"question": "Summarize the key takeaways in bullet points."})
                    st.subheader("Actionable Takeaways")
                    st.write(takeaways["answer"])
else:
    st.info("👈 Please upload a PDF document to get started")
    
    st.header("Sample Questions You Can Ask")
    st.write("""
    Once you upload a document, you can ask questions like:
    - What are the main topics covered in this document?
    - Explain the concept of X mentioned in the document.
    - How does this document address Y problem?
    - What are the author's conclusions about Z?
    """)
