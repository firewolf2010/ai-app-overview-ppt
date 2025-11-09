# minimal_ai_app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
generator = pipeline("text-generation", model="distilgpt2")  # 小模型示例

class Req(BaseModel):
    prompt: str
    max_length: int = 50

@app.post("/generate")
async def generate(req: Req):
    outputs = generator(req.prompt, max_length=req.max_length, num_return_sequences=1)
    return {"generated_text": outputs[0]["generated_text"]}