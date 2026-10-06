import json

from openai import OpenAI

from shopwright.config import LLM_API_KEY, LLM_BASE_URL, LLM_EXTRA_BODY, LLM_MODEL

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

tools = [{
    "type": "function",
    "function": {
        "name": "search_products",
        "description": "Search the electronics catalog with a free-text query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What the shopper wants, in plain words"},
                "max_price": {"type": "number", "description": "Optional upper price limit in USD"},
            },
            "required": ["query"],
        },
    },
}]

def ask(messages):
    r = client.chat.completions.create(model=LLM_MODEL, messages=messages,
                                       tools=tools, extra_body=LLM_EXTRA_BODY)
    return r.choices[0].message

messages = [{"role": "user", "content": "Find me noise cancelling earbuds under $50"}]
msg = ask(messages)
print("TURN 1 content:", msg.content)
for tc in msg.tool_calls or []:
    print("TURN 1 tool call:", tc.function.name, tc.function.arguments)

messages.append(msg.model_dump(exclude_none=True))
for tc in msg.tool_calls or []:
    fake = [{"title": "TREBLAB XR700 Running Earbuds", "price": 39.97, "rating": 4.3},
            {"title": "FUNSOUND ANC Earbuds", "price": 23.99, "rating": 4.1}]
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(fake)})

print("\nTURN 2 answer:", ask(messages).content)
