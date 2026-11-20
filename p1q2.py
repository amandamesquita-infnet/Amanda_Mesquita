from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Cria a aplicação FastAPI
app = FastAPI(
    title="API de Tradução Inglês-Francês",
    description="API para traduzir textos do inglês para o francês usando Hugging Face.",
    version="1.0.0",
)


# Define o formato do JSON recebido
class TranslationRequest(BaseModel):
    text: str


# Define o formato da resposta
class TranslationResponse(BaseModel):
    translated_text: str


# Nome do modelo utilizado
MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"


# Carrega o tokenizador e o modelo uma única vez
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


@app.get("/")
def home():
    return {
        "message": "API de tradução inglês-francês funcionando."
    }


@app.post("/translate", response_model=TranslationResponse)
def translate_text(request: TranslationRequest):
    try:
        # Converte o texto recebido em tokens
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # Gera a tradução
        output = model.generate(
            **inputs,
            max_new_tokens=512,
        )

        # Converte os tokens gerados novamente para texto
        translated_text = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        return {
            "translated_text": translated_text
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Erro ao traduzir o texto."
        ) from error