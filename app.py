from flask import Flask, render_template, request, jsonify
import os
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
RETRIEVAL_CHAIN = None

def initialize_rag():
    global RETRIEVAL_CHAIN
    if not os.path.exists("jd.pdf"): return False
    try:
        loader = PyPDFLoader("jd.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # 2026 Standard: Use LangChain Chroma & OllamaLLM
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        llm = OllamaLLM(model="llama3.2",max_tokens=200)
        prompt = ChatPromptTemplate.from_template("""
    You are a concise technical assistant. 
    Answer the following question based ONLY on the provided context.
    If the answer is not in the context, say "I don't know."
    Do not repeat the question. Do not be wordy.

    Context:
    {context}

    Question:
    {input}
    
    Answer:
""")
        
        # New chain structure
        combine_docs = create_stuff_documents_chain(llm, prompt)
        RETRIEVAL_CHAIN = create_retrieval_chain(retriever, combine_docs)
        return True
    except Exception as e:
        print(f"CRITICAL ERROR: {e}") # This will show the real error in your terminal
        return False

@app.route('/')
def home(): return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    if RETRIEVAL_CHAIN is None and not initialize_rag():
        return jsonify({"answer": "Engine failed to boot."})
    msg = request.json.get('message')
    res = RETRIEVAL_CHAIN.invoke({"input": msg})
    return jsonify({"answer": res["answer"]})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)