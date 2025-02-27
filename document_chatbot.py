import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing. Check your .env file!")

pdf_loader = PyPDFLoader("docker_article.pdf")
documents = pdf_loader.load()

if not documents:
    raise ValueError("❌ No text extracted from the document! Check the PDF file.")

print(f"✅ First 2 pages content preview: {documents[:2]}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)

text_chunks = text_splitter.split_documents(documents)

if not text_chunks:
    raise ValueError("❌ No text chunks available. Check text extraction.")

print(f"✅ Number of text chunks: {len(text_chunks)}")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

vector_store = FAISS.from_documents(text_chunks, embeddings)

vector_store.save_local("faiss_index")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.3,
    max_tokens=1024,
    openai_api_key=openai_api_key
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query = "What are the key points discussed in the document?"
response = qa_chain.invoke({"query": query}) 
print(f"✅ Refined Response:\n{response}")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

print("\n🔹 Summary:")
print(qa_chain.invoke({"question": "Provide a concise summary of the document."})["answer"])  

print("\n🔹 Detailed Analysis:")
print(qa_chain.invoke({"question": "Explain the first section in detail."})["answer"])

print("\n🔹 Insights & Takeaways:")
print(qa_chain.invoke({"question": "What are the major insights from this document?"})["answer"])

print("\n🔹 Actionable Points:")
print(qa_chain.invoke({"question": "Summarize the key takeaways in bullet points."})["answer"])
