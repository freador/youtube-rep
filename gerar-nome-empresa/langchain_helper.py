from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
#openai_organization = os.getenv("OPENAI_ORGANIZATION")

def generate_company_name(segmento):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=openai_api_key
    )

    chat_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você é um assistente IA que sempre responde em português do brasil",
            ),
            (
                "human",
                "Gere 5 ideias de nomes para empresas no segmento {segmento}",
            ),
        ]
    )
    company_names_chain= LLMChain(
        llm=llm, prompt=chat_template, output_key="company_name"
    )
    response = company_names_chain({"segmento": segmento})
    
    return response

if __name__ == "__main__":
    print(generate_company_name("imobiliaria"))