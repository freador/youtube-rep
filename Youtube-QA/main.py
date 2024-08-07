# Uso básico da langchain com os chatmodels, da ChatOpenAI
# Human Mesage and System Message para fazer tanto o prompt quanto a mensagem do usuários
# Uso básico de vector stores usando LLMChain (para chama)
# o Prompt Template para conseguimos deixar dinâmico
# o banco vectorial para criamos aplicações com RAG mais baratas e locais
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS

# Documento Loader do youtube
# O text splitter para fazer os chuncks
# E o Embendgin da Open Ai para transformamos os textos em matrizes indexaveis

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain

from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key
)

def create_vector_from_yt_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url, language="pt")
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query)
    docs_page_content = " ".join([d.page_content for d in docs])
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=openai_api_key
    )

    chat_template = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                """Você é um assistente que responde perguntas sobre videos do youtube baseado na transcrição do video.

                Responda a seguinte pergunta: {pergunta}
                Procurando nas seguintes transcrições: {docs}

                Use somente informação da transcrição para responder a pergunta. se você não sabe, Responda com "Eu não sei".

                Suas respostas devem ser em 200 caracteres
                """
            )
        ]
    )

    chain = LLMChain(llm=llm, prompt=chat_template, output_key="answer")

    response = chain({"pergunta": query, "docs": docs_page_content})

    return response, docs


if __name__ == "__main__":
    db = create_vector_from_yt_url("https://www.youtube.com/watch?v=nERonZFIIcI")
    response, docs = get_response_from_query(
        db, "O que é falado no video?")
    
    print(response)