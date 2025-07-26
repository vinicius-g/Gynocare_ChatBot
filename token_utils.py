# token_utils.py
from transformers import AutoTokenizer
from langchain_core.messages import BaseMessage

# Carrega o tokenizador da LLaMA 3 usada pelo Groq
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
LIMITE_TOKENS_MODELO = 8000  # um pouco abaixo do limite real de 8192

def contar_tokens(texto: str) -> int:
    return len(tokenizer.tokenize(texto))

def truncar_historico(chat_history: list[BaseMessage], system_prompt: str, user_input: str, db_msgs: list[BaseMessage]) -> list[BaseMessage]:
    tokens_total = 0

    # Componentes fixos
    tokens_total += contar_tokens(system_prompt)
    tokens_total += contar_tokens(user_input)
    tokens_total += sum(contar_tokens(m.content) for m in db_msgs)

    # Calcular espaço restante
    tokens_restantes = LIMITE_TOKENS_MODELO - tokens_total
    if tokens_restantes <= 0:
        return []  # sem espaço para histórico

    # Truncar histórico (priorizando mensagens mais recentes)
    tokens_por_msg = [(contar_tokens(m.content), m) for m in chat_history]
    historico_truncado = []
    for n_tokens, msg in reversed(tokens_por_msg):
        if tokens_restantes - n_tokens >= 0:
            historico_truncado.insert(0, msg)
            tokens_restantes -= n_tokens
        else:
            break

    return historico_truncado
