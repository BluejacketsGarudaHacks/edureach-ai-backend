from google import genai
from google.genai import types
import os

class Summarizer:
    def __init__(self):
        super().__init__()
        self.API_KEY = os.getenv('GEMINI_API')
        self.client = genai.Client(api_key=self.API_KEY)

    def answer(self, prompt: str):
        model = "gemini-2.0-flash-lite"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="text/plain",
        )

        result = ""

        for chunk in self.client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            result += chunk.text
            # print(chunk.text, end="")
        
        return result
