from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.language_models.fake import FakeListLLM


# Cria a aplicação FastAPI
app = FastAPI(
    title="Chatbot com Fake LLM",
    description="Protótipo de chatbot utilizando LangChain e Fake LLM.",
    version="1.0.0",
)


# Define o formato do JSON recebido
class ChatRequest(BaseModel):
    question: str


# Define o formato da resposta
class ChatResponse(BaseModel):
    answer: str


# Perguntas e respostas pré-definidas
predefined_answers = {
    "O que é Python?": "Python é uma linguagem de programação.",
    "O que é FastAPI?": "FastAPI é um framework Python para criação de APIs.",
    "O que é LangChain?": "LangChain é um framework para aplicações com modelos de linguagem.",
}


@app.get("/")
def home():
    return {"message": "Chatbot Fake LLM funcionando."}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    answer = predefined_answers.get(
        request.question,
        "Desculpe, não tenho uma resposta pré-definida para essa pergunta.",
    )

    # FakeListLLM simula a resposta de um modelo real
    fake_llm = FakeListLLM(
        responses=[answer]
    )

    response = fake_llm.invoke(request.question)

    return {
        "answer": response
    }