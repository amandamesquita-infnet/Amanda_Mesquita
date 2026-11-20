from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from langchain_core.runnables import RunnableLambda


# Modelo exigido no enunciado
MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"


# Verifica se existe GPU disponível
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# Carrega o tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# Carrega o modelo de tradução
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model.to(device)


# Função responsável pela tradução com Hugging Face
def translate_with_huggingface(
    text: str
) -> str:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )

    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
        )

    translated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return translated_text


# LangChain gerencia a chamada ao modelo Hugging Face
translator = RunnableLambda(
    translate_with_huggingface
)


# Cria a aplicação FastAPI
app = FastAPI(
    title="Tradutor Inglês-Alemão com LangChain e Hugging Face",
    description=(
        "API para traduzir textos do inglês para o alemão "
        "utilizando LangChain e o modelo "
        "Helsinki-NLP/opus-mt-en-de."
    ),
    version="1.0.0",
)


# Estrutura do JSON recebido
class TranslationRequest(BaseModel):
    text: str


# Estrutura do JSON retornado
class TranslationResponse(BaseModel):
    translated_text: str


@app.get("/")
def home():
    return {
        "message": (
            "API de tradução inglês-alemão funcionando."
        ),
        "model": MODEL_NAME,
        "device": device,
    }


@app.post(
    "/translate",
    response_model=TranslationResponse,
)
def translate_text(
    request: TranslationRequest
):
    try:
        # LangChain executa a função de tradução
        result = translator.invoke(
            request.text
        )

        return {
            "translated_text": result
        }

    except Exception as error:
        print(
            "ERRO REAL:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail="Erro ao realizar a tradução.",
        ) from error