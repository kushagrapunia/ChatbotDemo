import os
from flask import Flask, request
from twilio.rest import Client
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from PyPDF2 import PdfReader
import config


def delete_directory_contents(directory_path):
    # Verify if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # Iterate through all items in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        # Check if it's a file and delete it
        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Deleted file: {item_path}")

        # Check if it's a directory and delete its contents recursively
        elif os.path.isdir(item_path):
            delete_directory_contents(item_path)
                    
def create_index(file_path: str) -> None:
    reader = PdfReader(file_path)
    text = ''.join(page.extract_text() for page in reader.pages)
    with open(f'{config.OUTPUT_DIR}/output.txt', 'w') as file:
        file.write(text)

    loader = DirectoryLoader(
        config.OUTPUT_DIR,
        glob='**/*.txt',
        loader_cls=TextLoader
    )

    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1024,
        chunk_overlap=128
    )

    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        openai_api_key=config.OPENAI_API_KEY
    )

    persist_directory = config.DB_DIR

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectordb.persist()

def create_conversation() -> ConversationalRetrievalChain:

    persist_directory = config.DB_DIR

    embeddings = OpenAIEmbeddings(
        openai_api_key=config.OPENAI_API_KEY
    )

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=False
    )

    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(),
        chain_type='stuff',
        retriever=db.as_retriever(),
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True,
    )

def send_message(to: str, message: str) -> None:
    '''
    Send message to a Whatsapp user.
    Parameters:
        - to(str): sender whatsapp number in this whatsapp:+919558515995 form
        - message(str): text message to send
    Returns:
        - None
    '''
    account_sid = config.TWILIO_SID
    auth_token = config.TWILIO_TOKEN
    client = Client(account_sid, auth_token)
    _ = client.messages.create(from_=config.FROM, body=message, to=to)


# delete_directory_contents("data/db")
# delete_directory_contents("data/output")

create_index('data/input/sample.pdf')

qa = create_conversation()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return 'OK', 200

@app.route('/twilio', methods=['POST'])
def twilio():
    query = request.form['Body']
    sender_id = request.form['From']
    print(sender_id, query)
    # TODO
    # get the user
    # if not create
    # create chat_history from the previous conversations
    # question and answer
    res = qa(
        {
        'question': query,
        'chat_history': {}
        }
    )

    print(res)
    
    send_message(sender_id, res['answer'])

    return 'OK', 200