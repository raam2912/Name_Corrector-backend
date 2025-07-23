from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import datetime
import re

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma # Not strictly used in this specific flow, but common for LangChain setups
from langchain_community.document_loaders import TextLoader # Not strictly used in this specific flow
from langchain.text_splitter import CharacterTextSplitter # Not strictly used in this specific flow
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Not strictly used in this specific flow
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Load environment variables from .env file (for local testing)
# On Render, environment variables are set directly in the Render dashboard.
load_dotenv()

app = Flask(__name__)

# --- CORS Configuration ---
# IMPORTANT: This allows requests from your React frontend URL.
# Replace 'https://your-github-pages-username.github.io' with your actual GitHub Pages domain.
# For local testing, add 'http://localhost:3000' if your React app runs there.
CORS(app, resources={r"/*": {"origins": ["https://namecorrectionsheelaa.netlify.app","https://your-github-pages-username.github.io", "http://localhost:3000"]}})


# --- Global Variables for LLM, Vector Store, Memory, and Agent ---
llm = None
vectorstore = None # Not strictly used in this specific flow
memory = None
agent_executor = None
# On Render, GOOGLE_API_KEY will be pulled from environment variables.
# For local testing, it comes from .env.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Numerology Calculator Tool Implementation ---
# This is the same logic as in the frontend, but callable by the LLM on the backend.
NUMEROLOGY_MAP = {
    'A': 1, 'J': 1, 'S': 1,
    'B': 2, 'K': 2, 'T': 2,
    'C': 3, 'L': 3, 'U': 3,
    'D': 4, 'M': 4, 'V': 4,
    'E': 5, 'N': 5, 'W': 5,
    'F': 6, 'O': 6, 'X': 6,
    'G': 7, 'P': 7, 'Y': 7,
    'H': 8, 'Q': 8, 'Z': 8,
    'I': 9, 'R': 9
}

def reduce_number(num: int, allow_master_numbers: bool = False) -> int:
    # Corrected Python comparison operator from '===' to '=='
    if allow_master_numbers and (num == 11 or num == 22 or num == 33):
        return num
    while num > 9:
        if allow_master_numbers and (num == 11 or num == 22 or num == 33):
            break
        num = sum(int(digit) for digit in str(num))
    return num

def calculate_name_number(name: str) -> int:
    total = 0
    cleaned_name = ''.join(filter(str.isalpha, name)).upper()
    for letter in cleaned_name:
        if letter in NUMEROLOGY_MAP:
            total += NUMEROLOGY_MAP[letter]
    return reduce_number(total, False)

def calculate_life_path_number(birth_date_str: str) -> int:
    try:
        year, month, day = map(int, birth_date_str.split('-'))
    except ValueError:
        raise ValueError("Invalid birth date format. Please use YYYY-MM-DD.")

    month = reduce_number(month, True)
    day = reduce_number(day, True)
    year = reduce_number(year, True)

    total = month + day + year
    return reduce_number(total, True)

def numerology_calculator_tool_func(full_name: str, birth_date: str) -> str:
    """
    Calculates the Expression/Destiny Number from a full name and the Life Path Number from a birth date.
    This tool is useful for understanding a person's core numerological profile.
    It requires two arguments:
    - `full_name`: The person's full name (string, e.g., "John Doe Smith").
    - `birth_date`: The person's birth date in YYYY-MM-DD format (string, e.g., "1990-01-15").
    The tool will return a string detailing the calculated numerology numbers or an error message.
    Use this tool when the user asks about their numerology, or asks for a name correction based on numerology.
    """
    print(f"\n--- NUMEROLOGY CALCULATION REQUEST ---")
    print(f"Name: {full_name}")
    print(f"Birth Date: {birth_date}")
    print(f"--------------------------------------\n")

    if not full_name or not birth_date:
        return "Please provide both the full name and birth date (YYYY-MM-DD) to calculate numerology."

    try:
        expression_number = calculate_name_number(full_name)
        life_path_number = calculate_life_path_number(birth_date)
        return (f"For the name '{full_name}', the Expression/Destiny Number is {expression_number}. "
                f"For the birth date '{birth_date}', the Life Path Number is {life_path_number}.")
    except ValueError as e:
        return f"Error calculating numerology: {e}. Please ensure the date is in YYYY-MM-DD format."
    except Exception as e:
        return f"An unexpected error occurred during numerology calculation: {e}"


# --- Define Langchain Tools ---
tools = [
    Tool(
        name="numerology_calculator",
        func=numerology_calculator_tool_func, # Use the defined function
        description="""
        Calculates the Expression/Destiny Number from a full name and the Life Path Number from a birth date.
        This tool is useful for understanding a person's core numerological profile.
        It requires two arguments:
        - `full_name`: The person's full name (string, e.g., "John Doe Smith").
        - `birth_date`: The person's birth date in YYYY-MM-DD format (string, e.g., "1990-01-15").
        The tool will return a string indicating the calculated Expression Number and Life Path Number.
        Use this tool when the user asks about their numerology, or asks for a name correction based on numerology.
        """
    )
    # You can add other tools here if needed, like a scheduling tool, etc.
]

# --- Agent Prompt Template ---
# This prompt is designed to guide the LLM in name correction logic.
AGENT_PROMPT_TEMPLATE = """
You are Sheelaa's Elite AI Assistant - a warm, intuitive spiritual guide with 45+ million lives transformed.
Your primary role is to provide numerology name corrections and insights.
Respond with genuine warmth, ancient wisdom, and focused clarity.

**YOUR ESSENCE:**
- **Warmly Intuitive:** You sense deeper meanings, respond with empathy.
- **Confidently Wise:** You share knowledge from transforming 45+ million lives.
- **Genuinely Caring:** You ask one thoughtful follow-up that opens deeper discovery.
- **Encouragingly Authentic:** You celebrate progress, guide gently toward transformation.

**STRICT ADHERENCE TO CONTEXT & TOOLS:**
Your responses MUST be derived EXCLUSIVELY from the output of the tools you use.
NEVER introduce external information, personal opinions, assumptions, or fabricated details.
When generating name suggestions, ensure they are acceptable, usable, and sound natural. Avoid nonsensical or overly abstract names.

**Chat History:** {chat_history}
**Agent Scratchpad:**
{agent_scratchpad} # This is where the agent writes its thoughts and tool calls

**User Query: {input}**

---

**ENHANCED RESPONSE PROTOCOL:**

**1. INTENT RECOGNITION:**
    - The user's primary intent will be to get a **name correction based on numerology**.
    - You will be provided with their full name, birth date, and desired outcome in the `User Query`.

**2. LEAD WITH WARMTH & UNDERSTANDING:**
    - Acknowledge the person's situation with genuine care.
    - Show you understand why they're seeking guidance.

**3. PROVIDE COMPREHENSIVE, CONTEXTUAL ANSWERS (for name correction requests):**
    - **Step 1: Get Current Numerology.** The `User Query` will contain the full name, birth date, current Expression Number, and Life Path Number. Acknowledge these.
    - **Step 2: Interpret Desired Outcome.** Based on the user's "desired outcome" (e.g., "more success", "better relationships", "inner peace"), infer the most suitable target numerology number(s) for their name.
        - *Guidance for LLM:*
            - **1 (Leadership, New Beginnings, Drive):** For success, ambition, starting new ventures.
            - **2 (Cooperation, Balance, Harmony):** For relationships, diplomacy, peace.
            - **3 (Creativity, Expression, Joy):** For communication, artistic pursuits, optimism.
            - **4 (Stability, Structure, Hard Work):** For security, building foundations, discipline.
            - **5 (Freedom, Change, Adventure):** For adaptability, travel, dynamic life.
            - **6 (Responsibility, Nurturing, Love):** For family, service, community, healing.
            - **7 (Spirituality, Introspection, Wisdom):** For inner growth, analysis, research.
            - **8 (Abundance, Power, Material Success):** For financial gain, business acumen, leadership.
            - **9 (Humanitarianism, Compassion, Completion):** For selfless service, ending cycles, universal love.
            - **11 (Master Intuition):** For spiritual insight, inspiring others (often for 2-related goals).
            - **22 (Master Builder):** For large-scale manifestation, practical idealism (often for 4-related goals).
            - **33 (Master Healer/Teacher):** For compassionate service, universal love (often for 6-related goals).
    - **Step 3: Brainstorm Sensible Name Variations.** Generate **at least 6** new, acceptable, and usable full name suggestions (e.g., subtle spelling changes, adding or changing a middle name, or suggesting an entirely new first name if appropriate and common).
        - **Crucial:** For each brainstormed name, you must internally determine its Expression/Destiny Number by applying the numerology calculation rules. You can simulate calling the `numerology_calculator` tool with the new name to verify its numerology.
        - **Focus on names that sound natural and are culturally appropriate.** Avoid random letter combinations or nonsensical spellings.
    - **Step 4: Select and Explain Best Names.** Present the **6 or more** best name suggestions. For each suggestion:
        - State the suggested name clearly.
        - State its calculated Expression/Destiny Number.
        - Explain *how* this new number's energy aligns with the user's desired outcome, drawing from numerological interpretations.
        - Explain why it's a beneficial change.

    **Format your response clearly, using Markdown.**
    Start with a warm, empathetic opening.
    Then, present the current numerology and its explanation.
    Follow with a section for "Suggested Name Corrections" with each suggestion as a bullet point or numbered list item.
    For each suggestion, use the format:
    **Suggested Name:** [The new name]
    **Expression Number:** [Calculated number]
    **Explanation:** [Detailed explanation of alignment with desired outcome]

    ... and so on for at least 6 suggestions.

    **Finally, conclude your response with this exact sentence:**
    "For a much detailed report, book your appointment using Sheelaa.com."

    Ensure the explanations are encouraging and insightful, reflecting Sheelaa's wisdom.

**4. ASK MEANINGFUL FOLLOW-UPS (if more info is needed for tool or conversation):**
    - If any details are missing from the user's initial request for name correction (e.g., full name, birth date, or desired outcome), you MUST ask for them clearly. Example: "To provide a personalized name correction, I'll need your full name, your birth date in YYYY-MM-DD format, and a brief description of the positive outcome you desire in your life. Could you please share these details?"

**5. BRAND VOICE - SHEELAA'S WISDOM:**
    - Speak with the authority of someone who has guided millions.
    - Use language that reflects spiritual wisdom: "alignment," "harmony," "life path," "divine timing."
    - Share the confidence that comes from 99% client satisfaction.
    - Balance ancient wisdom with practical modern guidance.

**6. IMPACT-FOCUSED WRITING:**
    - Lead with transformation, not process.
    - Use powerful, decisive language: "reveals," "unlocks," "transforms," "illuminates."
    - Focus on the end result they'll experience.
    - Be specific about what they'll discover or achieve.
    - Cut filler words and get straight to the value.

Remember: Create connection and clarity in fewer words. Every sentence should move them closer to transformation.
"""

# Create a PromptTemplate instance for the agent
# 'context' is removed from input_variables as we're focusing on direct tool use
# and LLM generation for name correction, not RAG from a knowledge base for this specific flow.
AGENT_PROMPT = PromptTemplate(
    template=AGENT_PROMPT_TEMPLATE,
    input_variables=["chat_history", "input", "agent_scratchpad"]
)

# --- Helper function to format chat history ---
def format_chat_history(memory_instance):
    """
    Format the chat history from memory into a readable string format for the prompt.
    """
    try:
        messages = memory_instance.chat_memory.messages
        if not messages:
            return "No previous conversation."

        formatted_history = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_history.append(f"User: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"Assistant: {message.content}")

        return "\n".join(formatted_history)
    except Exception as e:
        print(f"Error formatting chat history: {e}")
        return "No previous conversation."


# --- Function to Initialize LLM and Agent ---
def initialize_llm_and_agent():
    global llm, memory, agent_executor

    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        return False

    try:
        # Initialize the LLM with tools
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Using a powerful model for better name generation
            google_api_key=GOOGLE_API_KEY,
            temperature=0.8, # Higher temperature for more creativity in name suggestions
            top_p=0.9,
            top_k=40
        )
        print("LLM (Gemini) initialized successfully.")

        # Initialize conversational memory
        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
        print("Conversational memory initialized.")

        # Define the agent
        agent = create_tool_calling_agent(
            llm=llm,
            tools=tools,
            prompt=AGENT_PROMPT
        )

        # Create the AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True # This helps the agent recover from malformed tool calls
        )
        print("Agent Executor initialized successfully.")

        return True

    except Exception as e:
        print(f"Error initializing LLM and agent: {e}")
        return False

# Initialize LLM and agent on app startup
if not initialize_llm_and_agent():
    print("Failed to initialize LLM and agent on startup. API will not function correctly.")

# --- Health Check Endpoint ---
@app.route("/health", methods=["GET"])
def health_check():
    """
    Endpoint to check the health and status of the backend service.
    """
    return jsonify({"status": "ok", "message": "Sheelaa Chatbot Backend is running!"}), 200

# --- Chat Endpoint ---
@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint to handle chat messages from the frontend.
    Receives a message, processes it using the LLM and agent,
    and returns a response.
    """
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    print(f"Received message: {user_message}")

    if agent_executor is None:
        print("Error: Agent Executor not initialized.")
        return jsonify({"error": "Chatbot not fully initialized. Please check backend logs."}), 500

    try:
        # The agent executor will handle tool calls and response generation
        result = agent_executor.invoke({"input": user_message})

        bot_response = result.get("output", "I apologize, but I couldn't process your request at this moment. Please try again or rephrase your query.")

        print(f"Sending response: {bot_response}")
        return jsonify({"response": bot_response}), 200

    except Exception as e:
        print(f"Error processing chat message: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An error occurred while processing your request. Please try again."}), 500

# This block ensures the Flask development server runs only when the script is executed directly
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
