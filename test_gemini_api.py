import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import asyncio

# Load environment variables from a .env file (if you have one locally)
load_dotenv()

async def test_gemini_connection():
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        print("Please set it before running this script (e.g., export GOOGLE_API_KEY='YOUR_KEY_HERE')")
        return

    print(f"Attempting to connect to Gemini with API Key status: {'[KEY FOUND]' if google_api_key else '[KEY NOT FOUND]'}")

    try:
        # Use the same model and settings as in your Flask app
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.0-pro",
            google_api_key=google_api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )

        print("LLM instance initialized. Sending a test message...")

        # Create a simple prompt
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="Hello, what is your purpose?")
        ]

        # Invoke the model
        response = await llm.ainvoke(messages)
        print("\n--- Gemini API Test Successful ---")
        print("Response from Gemini:")
        print(response.content)
        print("----------------------------------")

    except Exception as e:
        print("\n--- Gemini API Test FAILED ---")
        print(f"An error occurred: {e}")
        print("This likely indicates an issue with your API Key, its permissions, or model availability.")
        print("Please ensure:")
        print("1. Your GOOGLE_API_KEY is correct and active.")
        print("2. The Generative Language API is enabled in your Google Cloud Project.")
        print("3. Your API Key has permissions to use the gemini-1.0-pro model.")
        print("------------------------------")

if __name__ == "__main__":
    asyncio.run(test_gemini_connection())
