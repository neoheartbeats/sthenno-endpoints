from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def speech_to_filename(text: str, filename: str) -> None:
    """
    Convert TEXT to speech and save it to FILENAME.

    Args:
        filename (str): The name of the file to save the speech to.
        text (str): The text to convert to speech.
    """
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="echo",
        input=text,
    ) as response:
        response.stream_to_file(filename)
