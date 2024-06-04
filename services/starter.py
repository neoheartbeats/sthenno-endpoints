from functions import text_generation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import TypedDict
import json

app = FastAPI()


class TranslationRequest(BaseModel):
    content: str


class TranslationResponse(TypedDict):
    translation: str


@app.post("/translate_to_zh/")
async def translate_to_zh(input: TranslationRequest) -> TranslationResponse | None:
    translation_system_content: str = (
        "You are a helpful assistant designed to output JSON."
        "Translate user's input into Chinese in a JSON format like this:"
        "user's input: `evening`"
        "your output: `{'translation': '夜晚'}`"
    )

    try:
        translation = await text_generation.get_chat_completion_content(
            system_content=translation_system_content,
            user_content=input.content,
            using_type=True,
        )

        if translation is None:
            return None
        translation_content: str = json.loads(translation)["translation"]
        return TranslationResponse(translation=translation_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
