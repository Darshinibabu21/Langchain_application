# **AI-Powered Document Chatbot**  

## ** Overview**  
This project builds a **conversational AI** that can answer queries about a PDF document using **LangChain, FAISS, and OpenAI’s GPT-4**. The system:  
- **Processes a PDF document** and splits it into manageable chunks.  
- **Generates embeddings** and stores them in a FAISS vector database.  
- **Retrieves relevant content** using similarity search.  
- **Uses GPT-4 for answering queries** based on retrieved content.  
- **Maintains conversation history** for context-aware responses.  

---

## ** Required Files**  
- **`main.py`** → Main script to process the document and handle queries.  
- **`docker_article.pdf`** → The PDF document to analyze (place in the project folder).  
- **`.env`** → Stores the OpenAI API key.  
- **`requirements.txt`** → List of dependencies.  
- **`faiss_index/`** → Directory where FAISS vector database is stored.  

---

## **⚙️ Setup Instructions**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your-username/langchain-pdf-chatbot.git
cd langchain-pdf-chatbot
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Add Your OpenAI API Key**  
Create a `.env` file in the project directory and add:  
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### **4️⃣ Upload Your PDF File**  
Place **`docker_article.pdf`** in the project folder.  

### **5️⃣ Run the Script**  
```bash
python main.py
```

---
