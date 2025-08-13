from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

# This is the correct class for Gemini API
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
response = model.invoke("Hello Gemini, What can u do ?")
print(response.content)