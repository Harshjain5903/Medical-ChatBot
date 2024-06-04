from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
from src.pinecone_config import PineconeConfig  # Import the PineconeConfig
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Ensure that the API key is not None or empty
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set or is empty")

embeddings = download_hugging_face_embeddings()

# Build Pinecone configuration
config = PineconeConfig.build(api_key=PINECONE_API_KEY)

# Initialize Pinecone client with the built configuration
pc = pinecone.Pinecone(config=config)

# Loading the index
index_name = "medical-bot"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean',
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

docsearch = Pinecone.from_existing_index(index_name, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
