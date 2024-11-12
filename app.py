from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from transformers import pipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware


app = Flask(__name__)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Update with specific domains if needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
NVDIA_API_KEY=os.environ.get('NVDIA_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["NVDIA_API_KEY"] = NVDIA_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatNVIDIA(
  model="meta/llama-3.1-405b-instruct",
  api_key= NVDIA_API_KEY, 
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",device=0)

memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

def analyze_sentiment(user_input):
    result = sentiment_model(user_input)
    sentiment = result[0]['label']
    return sentiment


class SentimentAwareRAGPipeline:
    def __init__(self, rag_chain,memory):
        self.rag_chain = rag_chain
        self.memory = memory
    def run(self, user_input):
        # Step 1: Analyze the sentiment of user input
        sentiment = analyze_sentiment(user_input)
        
        # Step 2: Get chat history from memory
        chat_history = self.memory.load_memory_variables({})
        history_str = ""
        if "chat_history" in chat_history:
            # Convert chat history to string format
            for message in chat_history["chat_history"]:
                if isinstance(message, HumanMessage):
                    history_str += f"Human: {message.content}\n"
                elif isinstance(message, AIMessage):
                    history_str += f"Assistant: {message.content}\n"
        
        # Step 3: Modify the user input with sentiment and history information
        modified_input = f"""
        Chat History:
        {history_str}
        
        Sentiment: {sentiment}
        Current Query: {user_input}
        """
        
        # Step 4: Run the modified input through the RAG chain
        response = self.rag_chain.invoke({
            "input": modified_input
        })
        
        # Step 5: Save the interaction to memory
        self.memory.save_context(
            {"input": user_input},
            {"answer": response["answer"]}
        )
        
        return response
    

sentiment_aware_pipeline = SentimentAwareRAGPipeline(rag_chain,memory)


# class ChatRequest(BaseModel):
#     msg: str

@app.route("/")
def index():
    return render_template('chat.html')

# @app.post("/chat")
# async def chat(request: ChatRequest):
#     # Get the message from the request
#     user_input = request.msg
#     print(f"User Input: {user_input}")
    
#     # Get response from the sentiment-aware pipeline
#     response = sentiment_aware_pipeline.run(user_input)
    
#     # Return the response answer
#     return {"answer": response["answer"]}


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = sentiment_aware_pipeline.run(msg)
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)