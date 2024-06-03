from dotenv import load_dotenv
from openai import OpenAI
from typing import Final, Optional
import asyncio
import os

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def system_message(content: str) -> dict:
    return {"role": "system", "content": content}


def user_message(content: str) -> dict:
    return {"role": "user", "content": content}


def assistant_message(content: str) -> dict:
    return {"role": "assistant", "content": content}


GPT_MODEL: Final[str] = "gpt-4o"


async def get_chat_completion(system_content: str, user_content: str) -> dict | None:
    try:
        messages = [
            system_message(content=system_content),
            user_message(content=user_content),
        ]
        completion = client.chat.completions.create(
            model=GPT_MODEL,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0.0,
        )
        completion_content: str = completion.choices[0].message.content
        prompt_tokens: int = completion.usage.prompt_tokens
        completion_tokens: int = (
            completion.usage.total_tokens - completion.usage.prompt_tokens
        )

        return {
            "content": completion_content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


DEFAULT_SYSTEM_MESSAGE: Final[str] = "You are a helpful assistant."
DEFAULT_SYSTEM_MESSAGE_USING_TYPE: Final[str] = (
    "You are a helpful assistant designed to output JSON."
)


async def from_user_request(
    user_content: str,
    system_content: Optional[str] = DEFAULT_SYSTEM_MESSAGE_USING_TYPE,
) -> str | None:
    completion = await get_chat_completion(
        system_content=system_content,
        user_content=user_content,
    )
    if completion is not None:
        return completion.get("content")
    return None
