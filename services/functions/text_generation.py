from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_role import ChatCompletionRole
from typing import Iterable, Literal, TypedDict, TypeAlias, Any
import os

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class ChatCompletionMessageParam(TypedDict):
    role: ChatCompletionRole
    content: str


Message: TypeAlias = ChatCompletionMessageParam
Messages: TypeAlias = Iterable[Message]


def message(role: ChatCompletionRole, content: str) -> Message:
    return Message(role=role, content=content)


system_message = lambda content: message("system", content)
user_message = lambda content: message("user", content)
assistant_message = lambda content: message("assistant", content)


class ChatCompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int


class ChatCompletionParam(TypedDict):
    content: str | None
    usage: ChatCompletionUsage


class ResponseFormat(TypedDict, total=False):
    type: Literal["text", "json_object"]


# Using typing Any here for messages is not helpful. Messsages was
# used but error at client.chat.completions.create(**options) was
# reported by Pyright. This might be a bug of Pyright.


class CompletionCreateParam(TypedDict, total=False):
    model: str
    messages: Any
    temperature: float
    response_format: ResponseFormat


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
        options = CompletionCreateParam(
            model=gpt_model,
            messages=messages,
            temperature=0.0,
        )
        if using_type:
            options["response_format"] = ResponseFormat(type="json_object")
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
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        )

    except Exception as e:
        print(e)
        return None


async def get_chat_completion_content(
    user_content: str,
    system_content: str = "",
    using_type: bool = False,
) -> str | None:
    if system_content == "":
        if using_type:
            system_content = "You are a helpful assistant designed to output JSON."
        else:
            system_content = "You are a helpful assistant."
    completion: ChatCompletionParam | None = await get_chat_completion_param(
        system_content=system_content,
        user_content=user_content,
        using_type=using_type,
    )
    if completion is None:
        return None
    return completion["content"]


__all__ = ["get_chat_completion_content"]
