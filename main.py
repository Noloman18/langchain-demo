from dotenv import load_dotenv
from flask import Flask, request
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.faiss import FAISS


# Press the green button in the gutter to run the script.

def loadVectorStore():
    loader = DirectoryLoader("D:\OpenAI\MyData", glob="**/*.txt", show_progress=True)
    documents = loader.load()

    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)

def loadModel():
    llm = ChatOpenAI()
    llm.model_name = "gpt-3.5-turbo-16k"
    llm.temperature = 0

    return llm

def loadPrompt():
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    return ChatPromptTemplate.from_template(template)

load_dotenv()

db = loadVectorStore()

model = loadModel()

prompt = loadPrompt()

chain = (
        {"context": db.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

app = Flask(__name__)

@app.route('/query-my-data', methods=['POST'])
def post():
    data = request.get_json()
    return chain.invoke(data['question'])

if __name__ == '__main__':
    app.run(debug=True)
