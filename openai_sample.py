
from openai import OpenAI
from dotenv import load_dotenv,find_dotenv
import os
load_dotenv(find_dotenv())

client=OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


resp = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {"role": "system", "content": "You are concise and helpful."},
        {"role": "user", "content": "List 5 iconic Singapore hawker dishes, Where is lau pa sat"},
    ],
)
print(resp.choices[0].message.content)
