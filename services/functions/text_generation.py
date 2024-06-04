from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_role import ChatCompletionRole
from typing import Final, Iterable, TypedDict, TypeAlias
import os

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


Role: TypeAlias = ChatCompletionRole


class ChatCompletionMessageParam(TypedDict):
    role: Role
    content: str


Message: TypeAlias = ChatCompletionMessageParam
Messages: TypeAlias = Iterable[ChatCompletionMessageParam]


def message(role: Role, content: str) -> Message:
    return Message(role=role, content=content)


system_message = lambda content: message("system", content)
user_message = lambda content: message("user", content)
assistant_message = lambda content: message("assistant", content)


class ChatCompletionParam(TypedDict):
    content: str | None
    usage: dict[str, int]


async def get_chat_completion_param(
    system_content: str,
    user_content: str,
    using_type: bool = False,
) -> ChatCompletionParam | None:
    def create_completion(using_type: bool) -> ChatCompletion:
        messages: Messages = [
            system_message(system_content),
            user_message(user_content),
        ]
        gpt_model: str = "gpt-4o"
        options: dict = {
            "model": gpt_model,
            "messages": messages,
            "temperature": 0.0,
        }
        if using_type:
            options["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**options)

    try:
        completion: ChatCompletion = create_completion(using_type)
        completion_content: str | None = completion.choices[0].message.content

        if completion.usage is None:
            return None
        prompt_tokens: int = completion.usage.prompt_tokens
        completion_tokens: int = completion.usage.total_tokens - prompt_tokens

        return ChatCompletionParam(
            content=completion_content,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

    except Exception as e:
        print(e)
        return None


async def get_chat_completion_content(
    user_content: str,
    system_content: str = "",
    using_type: bool = False,
) -> str | None:
    DEFAULT_SYSTEM_MESSAGE: Final[str] = "You are a helpful assistant."
    DEFAULT_SYSTEM_MESSAGE_USING_TYPE: Final[str] = (
        "You are a helpful assistant designed to output JSON."
    )
    if system_content == "" and not using_type:
        system_content = DEFAULT_SYSTEM_MESSAGE
    elif system_content == "" and using_type:
        system_content = DEFAULT_SYSTEM_MESSAGE_USING_TYPE
    completion = await get_chat_completion_param(
        system_content=system_content,
        user_content=user_content,
        using_type=using_type,
    )
    if completion is None:
        return None
    return completion["content"]


__all__ = ["get_chat_completion_content"]
