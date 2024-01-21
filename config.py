import os
import tempfile

from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from openai import OpenAI

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME')

TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_TOKEN = os.getenv('TWILIO_TOKEN')
FROM = os.getenv('FROM')

CONNECTION_STRING = os.getenv('CONNECTION_STRING')
DATABASE_NAME = os.getenv('DATABASE_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

NUMBER_OF_DOCUMENTS = 6

QDRANT_COLLECTION_NAME = 'document_gpt'
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

embedding_function = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY
)

ERROR_MESSAGE = 'We are facing a techincal issue at this time, please try after sometime.'

OUTPUT_DIR = os.path.join(
    tempfile.gettempdir(),
    'document_gpt'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

chat_model = ChatOpenAI(
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
    model_name=OPENAI_MODEL_NAME
)

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)
