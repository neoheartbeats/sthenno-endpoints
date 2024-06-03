from functions import text_generation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json

app = FastAPI()


class TranslationRequest(BaseModel):
    content: str


@app.post("/translate/")
async def translate(input: TranslationRequest):
    translator_system_prompt: str = """You are a helpful assistant designed to output JSON.
Translate the user's input from English into Chinese in a JSON format like this:
input: `Hello`
output: `{"translation": "你好"}`
"""
    try:
        translation = await text_generation.from_user_request(
            system_content=translator_system_prompt,
            user_content=input.content,
        )
        translation_obj: dict = json.loads(translation)
        translation_content: str = translation_obj["translation"]
        return {"translation": translation_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
