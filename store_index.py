from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Load and preprocess PDF data
extracted_data = load_pdf_file(data="data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone Vector Store
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Upload documents in batches
batch_size = 50
for i in range(0, len(text_chunks), batch_size):
    batch = text_chunks[i:i + batch_size]
    vector_store.add_documents(batch)
    print(f"✅ Uploaded batch {i // batch_size + 1} / {(len(text_chunks) + batch_size - 1) // batch_size}")

print("🎉 All documents uploaded successfully!")
