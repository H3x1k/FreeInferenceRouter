from fastapi import FastAPI, HTTPException
from typing import Optional
from typing import Literal
from pydantic import BaseModel
from router import Router

app = FastAPI()
router = Router()


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class CompletionRequest(BaseModel):
    messages: list[Message]

class CompletionResponse(BaseModel):
    response: str


@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/generate", response_model=CompletionResponse)
def generate(request: CompletionRequest):
    try:
        response = router.generate(request.messages)
        return CompletionResponse(response=response)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
