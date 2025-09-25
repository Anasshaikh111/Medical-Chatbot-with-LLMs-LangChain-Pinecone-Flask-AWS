from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
index_name = "medical-chatbot"
if not pc.has_index(index_name):
    raise ValueError(f"Index '{index_name}' does not exist. Please run store_index.py first.")

# Load Hugging Face embeddings
embeddings = download_hugging_face_embeddings()

# Connect to Pinecone vector store
docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Setup retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Setup Hugging Face text-generation model
text_gen_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-small",  # You can replace with any HF model
    tokenizer="google/flan-t5-small",
    max_length=512
)
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# Setup prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Flask routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("Input:", msg)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
