from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai_core.assistant import ChatAssistant, BASE_SYSTEM_PROMPT, TOOLS, DEFAULT_MODEL

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


SESSIONS: dict[str, ChatAssistant] = {}


def get_assistant(session_id: str) -> ChatAssistant:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = ChatAssistant(
            system_prompt=BASE_SYSTEM_PROMPT,
            model=DEFAULT_MODEL,
            temperature=0.7,
            tools=TOOLS,
        )
    return SESSIONS[session_id]


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    assistant = get_assistant(req.session_id)
    reply = assistant.chat_step(req.message)
    return ChatResponse(reply=reply)
