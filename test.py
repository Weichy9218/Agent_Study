import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")

if not api_key:
    raise ValueError("Missing API key. Set OPENROUTER_API_KEY (or OPENAI_API_KEY) in .env.")

client_kwargs = {"api_key": api_key}
client_kwargs["base_url"] = base_url

client = OpenAI(**client_kwargs)

USER_PROMPT = r"""
You are an agent that can predict future events. The event to be predicted: \"2025-12-27, what will be the \"most recent\" number for \"Waiting for response\" of this date in the Support Stats published by Steam?\"\n        IMPORTANT: Your final answer MUST end with this exact format:\n        \\boxed{YOUR_PREDICTION}\n        Do not use any other format. Do not refuse to make a prediction. Do not say \"I cannot predict the future.\" You must make a clear prediction based on the best data currently available, using the box format specified above.
"""

resp = client.chat.completions.create(
    model="openai/gpt-5.2",
    messages=[
        {"role": "user", "content": USER_PROMPT},
    ],
    extra_body={
        "reasoning": {
            "effort": "medium"
        }
    }
)

content = resp.choices[0].message.content
reason_process = resp.choices[0].message.reasoning

print(content)
print("*" * 40)
print(reason_process)
