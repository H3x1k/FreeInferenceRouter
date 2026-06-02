from fastapi import FastAPI, HTTPException
from typing import Optional, Literal
from pydantic import BaseModel
from router import Router

app = FastAPI()
router = Router()


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict


class Tool(BaseModel):
    type: str = "function"
    function: FunctionDefinition


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None


class CompletionRequest(BaseModel):
    messages: list[Message]
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[str | dict] = None


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(request: CompletionRequest):
    try:
        kwargs = {}
        if request.tools:
            kwargs["tools"] = [t.model_dump() for t in request.tools]
        if request.tool_choice:
            kwargs["tool_choice"] = request.tool_choice
        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        response = router.generate(messages, **kwargs)
        if not response:
            raise HTTPException(status_code=503, detail="No available provider")
        return response
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
