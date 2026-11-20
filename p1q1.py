from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline


# Cria a aplicação FastAPI
app = FastAPI(
    title="API de Geração de Texto com GPT-2",
    description="API simples para gerar continuação de texto usando GPT-2.",
    version="1.0.0",
)


# Define o formato do JSON recebido
class TextRequest(BaseModel):
    text: str


# Define o formato da resposta
class TextResponse(BaseModel):
    generated_text: str


# Carrega o modelo GPT-2
generator = pipeline(
    "text-generation",
    model="openai-community/gpt2",
)


@app.get("/")
def home():
    return {"message": "API GPT-2 funcionando."}


@app.post("/generate", response_model=TextResponse)
def generate_text(request: TextRequest):
    try:
        result = generator(
            request.text,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            return_full_text=False,
            pad_token_id=generator.tokenizer.eos_token_id,
        )

        return {
            "generated_text": result[0]["generated_text"]
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Erro ao gerar o texto."
        ) from error