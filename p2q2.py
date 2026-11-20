from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# Carrega as variáveis do arquivo .env
load_dotenv()


# Cria a aplicação FastAPI
app = FastAPI(
    title="Tradutor Inglês-Francês com LangChain e Groq",
    description=(
        "API para traduzir textos do inglês para o francês "
        "usando LangChain e Groq."
    ),
    version="1.0.0",
)


# Define o formato do JSON recebido
class TranslationRequest(BaseModel):
    text: str


# Define o formato da resposta
class TranslationResponse(BaseModel):
    translated_text: str


# Configura o modelo acessado pelo Groq
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    timeout=20,
    max_retries=2,
)


# Define as instruções para a tradução
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Translate the user's English text into French. "
            "Return only the French translation."
        ),
        (
            "human",
            "{text}"
        ),
    ]
)


# Cria a cadeia LangChain
translator = prompt | model


@app.get("/")
def home():
    return {
        "message": (
            "API de tradução com LangChain e Groq funcionando."
        )
    }


@app.post(
    "/translate",
    response_model=TranslationResponse
)
def translate_text(
    request: TranslationRequest
):
    try:
        response = translator.invoke(
            {
                "text": request.text
            }
        )

        return {
            "translated_text": response.content
        }

    except Exception as error:
        print(
            "ERRO REAL:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail="Erro ao realizar a tradução."
        ) from error