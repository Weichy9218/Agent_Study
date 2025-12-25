import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env.")

client = OpenAI(api_key=api_key)

USER_PROMPT = r"""You are an agent that can predict future events...
The event to be predicted: "2025-12-25, what will be the total market capitalization (总市值) of TCL Technology (000100) on the Shenzhen Stock Exchange in hundreds of millions of yuan (two decimal places)?"
"""


class Pred(BaseModel):
    market_cap_hm_yuan: float = Field(..., description="单位：亿元，保留两位小数")
    reasoning_short: str


resp = client.responses.parse(
    model="gpt-5.2",
    tools=[{"type": "web_search"}],
    tool_choice="auto",
    include=["web_search_call.action.sources"],
    input=[
        {"role": "system", "content": "Use web search if helpful. Return structured output only."},
        {"role": "user", "content": USER_PROMPT},
    ],
    text_format=Pred,
)

pred = resp.output_parsed
print(f"\\boxed{{{pred.market_cap_hm_yuan:.2f}}}")
