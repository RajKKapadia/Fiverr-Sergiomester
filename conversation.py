from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.vectorstores.qdrant import Qdrant
from langchain.chains import LLMChain
from langchain.schema.document import Document

import config


def get_location(query: str) -> str:
    response_schemas = [
        ResponseSchema(
            name="location", description="it is a location."),
        ResponseSchema(
            name="quantity", description="it is a quantity and return only a number.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(
        response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(
                "try to extract a location and quantity from the question, if not found output -1.\n{format_instructions}\n{question}")
        ],
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions}
    )
    try:
        chat = ChatOpenAI(
            temperature=0, openai_api_key=config.OPENAI_API_KEY, openai_api_base='')
        _input = prompt.format_prompt(question=query)
        response = chat(_input.to_messages())
        output = output_parser.parse(response.content)
        return {
            'status': 1,
            'location': output['location'],
            'quantity': output['quantity']
        }
    except:
        return {
            'status': -1,
            'location': -1,
            'quantity': -1
        }


def format_context(documents: list[Document]) -> str:
    formated_context = ''
    for doc in documents:
        formated_context += f'\n{doc.page_content.strip()}\n'
    return formated_context


def format_chat_history(chat_history: list[list[str, str]]) -> str:
    chat_history = chat_history[:-1]
    formated_chat_history = ''
    for ch in chat_history:
        formated_chat_history += f'HUMAN: {ch[0]}\nAI: {ch[1]}\n'
    return formated_chat_history


def get_system_template() -> str:
    system_prompt = '''You are a helpful assistant. \
Use the following pieces of CONTEXT and CHAT HISTORY to answer the QUESTION at the end. \
If you don't know the answer and the CONTEXT doesn't contain the answer truthfully say I don't know. \
Keep an informative tone.'''
    instruction = "CONTEXT: {context}\n\nCHAT HISTORY:\n\n{chat_history}\n\nHUMAN: {question}\n\nAI:"
    template = f'{system_prompt}\n{instruction}'
    return template


def condense_user_query(query: str, chat_history: list[list]) -> tuple:
    system_prompt = '''Given the following CHAT HISTORY and a FOLLOW UP QUESTION, \
rephrase the FOLLOW UP QUESTION to be a STANDALONE QUESTION in its original language. \
Keep the context of the CHAT HISTORY in the standalone question.'''
    instruction = "CHAT HISTORY:\n\n{chat_history}\n\nFOLLOW UP QUESTION: {question}\n\nSTANDALONE QUESTION:"
    template = f'{system_prompt}\n{instruction}'
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(template)
        ]
    )
    if len(chat_history) <= 1:
        return query
    formated_chat_history = format_chat_history(chat_history)
    llm_chain = LLMChain(
        llm=config.chat_model,
        prompt=prompt,
        verbose=True
    )
    response = llm_chain.predict(
        question=query, chat_history=formated_chat_history)
    response = response.strip()
    return response


def create_llm_conversation(chat_history: list) -> list[list]:
    try:
        query = chat_history[-1][0]
        vector_db = Qdrant(client=config.qdrant_client, embeddings=config.embedding_function,
                           collection_name=config.QDRANT_COLLECTION_NAME)
        template = get_system_template()
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(template)
            ]
        )
        llm_chain = LLMChain(
            llm=config.chat_model,
            prompt=prompt,
            verbose=True
        )
        condense_query = condense_user_query(query, chat_history)
        searched_docs = vector_db.similarity_search(
            condense_query, k=config.NUMBER_OF_DOCUMENTS)
        formated_chat_history = format_chat_history(chat_history)
        formated_context = format_context(searched_docs)
        response = llm_chain.predict(
            question=query, context=formated_context, chat_history=formated_chat_history)
        response = response.strip()
        chat_history[-1][1] = response
        return chat_history
    except:
        chat_history.append((chat_history[-1][0], config.ERROR_MESSAGE))
        return chat_history


def handle_user_query(message: str, chat_history: list[tuple]) -> tuple:
    chat_history += [[message, None]]
    return '', chat_history


def format_chat_history_backend(chat_history: list[list[str, str]]) -> str:
    formated_chat_history = ''
    for ch in chat_history:
        formated_chat_history += f'HUMAN: {ch["query"]}\nAI: {ch["response"]}\n'
    return formated_chat_history


def create_llm_conversation_backend(chat_history: list[list], query: str) -> str:
    try:
        vector_db = Qdrant(client=config.qdrant_client, embeddings=config.embedding_function,
                           collection_name=config.QDRANT_COLLECTION_NAME)
        template = get_system_template()
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(template)
            ]
        )
        llm_chain = LLMChain(
            llm=config.chat_model,
            prompt=prompt,
            verbose=True
        )
        condense_query = condense_user_query(query, chat_history)
        searched_docs = vector_db.similarity_search(
            condense_query, k=config.NUMBER_OF_DOCUMENTS)
        formated_chat_history = format_chat_history_backend(chat_history)
        formated_context = format_context(searched_docs)
        response = llm_chain.predict(
            question=query, context=formated_context, chat_history=formated_chat_history)
        response = response.strip()
        return response
    except:
        config.ERROR_MESSAGE
