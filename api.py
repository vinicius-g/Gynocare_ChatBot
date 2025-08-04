from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from twilio.rest import Client
import os
from dotenv import load_dotenv
from chain import get_database_responses, create_chain
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

app = FastAPI()

# Twilio config
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = "whatsapp:+14155238886"  # número do sandbox
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Chatbot
chatbot_chain = create_chain()

class ChatRequest(BaseModel):
    user_input: str
    chat_history: list[dict]  # não é usado no WhatsApp, apenas no chat web

@app.post("/chat")
def responder_chat(req: ChatRequest):
    db_msgs, reasoning = get_database_responses(
        pergunta_usuario=req.user_input,
        chat_history=[
            HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
            for m in req.chat_history
        ],
        nome_colecao=os.getenv("NOME_COLECAO_DB", "qa_excel_collection"),
        diretorio_db=os.getenv("DIRETORIO_DB")
    )

    resposta = chatbot_chain.invoke({
        "user_input": req.user_input,
        "chat_history": [],
        "database_responses": db_msgs
    })

    return {"response": resposta, "reasoning": reasoning}

@app.post("/whatsapp")
async def receber_whatsapp(request: Request):
    form = await request.form()
    mensagem_usuario = form.get("Body")
    numero_origem = form.get("From")

    if not mensagem_usuario or not numero_origem:
        return PlainTextResponse("Faltando parâmetros", status_code=400)

    # Rodar o pipeline do chatbot
    db_msgs, _ = get_database_responses(
        pergunta_usuario=mensagem_usuario,
        chat_history=[],
        nome_colecao=os.getenv("NOME_COLECAO_DB", "qa_excel_collection"),
        diretorio_db=os.getenv("DIRETORIO_DB")
    )

    resposta = chatbot_chain.invoke({
        "user_input": mensagem_usuario,
        "chat_history": [],
        "database_responses": db_msgs
    })

    client.messages.create(
        from_=TWILIO_WHATSAPP_FROM,
        to= numero_origem,
        body=resposta
    )

    return ""
