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
CORS(app, resources={r"/*": {"origins": [
    "https://namecorrectionsheelaa.netlify.app", # Your Netlify frontend domain
    "http://localhost:3000" # For local development
]}})


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

# Interpretations for the LLM to use when explaining numbers
NUMEROLOGY_INTERPRETATIONS = {
    1: "Leadership, independence, new beginnings, drive, and ambition. It empowers you to forge your own path and initiate new ventures with confidence. Number 1 vibrates with pioneering spirit, self-reliance, and the courage to stand alone. It signifies a strong will, determination, and the ability to lead. Individuals with a strong 1 influence are often innovators, driven to achieve and create their own destiny. They thrive in environments where they can take charge and express their individuality.",
    2: "Cooperation, balance, diplomacy, harmony, and partnership. It fosters strong relationships, intuition, and a gentle, supportive nature. Number 2 is the peacemaker, symbolizing duality, grace, and tact. It represents sensitivity, empathy, and the ability to work well with others. Those influenced by 2 are often mediators, seeking equilibrium and understanding. They excel in collaborative efforts and bring a calming presence to any situation.",
    3: "Creativity, self-expression, communication, optimism, and joy. It enhances social interactions, artistic pursuits, and a vibrant outlook on life. Number 3 is the number of expression, inspiration, and growth. It signifies imagination, enthusiasm, and a natural talent for communication. Individuals with a strong 3 influence are often charismatic, artistic, and enjoy being in the spotlight. They bring light and joy to others through their creative endeavors and optimistic spirit.",
    4: "Stability, diligent hard work, discipline, organization, and building strong foundations for lasting security. It signifies reliability and a practical approach. Number 4 is the builder, representing order, structure, and practicality. It is associated with responsibility, honesty, and a strong work ethic. Those influenced by 4 are often methodical, reliable, and persistent. They excel at creating systems, managing resources, and ensuring long-term security through diligent effort.",
    5: "Freedom, dynamic change, adventure, versatility, and adaptability. It encourages embracing new experiences, travel, and a love for personal liberty. Number 5 is the number of change, symbolizing curiosity, progress, and a desire for new horizons. It represents adaptability, resourcefulness, and a love for exploration. Individuals with a strong 5 influence are often restless, seeking variety and excitement. They thrive on challenges and are quick to embrace new opportunities, often leading to diverse life experiences.",
    6: "Responsibility, nurturing, harmony, selfless service, and love. It fosters deep connections in family and community, embodying care and compassion. Number 6 is the number of harmony and domesticity, symbolizing love, empathy, and service to others. It is associated with responsibility, protection, and a strong sense of community. Those influenced by 6 are often caregivers, dedicated to their family and friends. They find fulfillment in supporting others and creating a peaceful, loving environment.",
    7: "Spirituality, deep introspection, analytical thought, wisdom, and inner truth. It encourages seeking knowledge, solitude, and understanding the deeper mysteries of life. Number 7 is the seeker, representing analysis, contemplation, and spiritual awareness. It signifies intuition, wisdom, and a desire for deeper understanding. Individuals with a strong 7 influence are often introspective, philosophical, and drawn to spiritual or intellectual pursuits. They prefer solitude for reflection and possess a keen ability to uncover hidden truths.",
    8: "Abundance, power, material success, leadership, and executive ability. It signifies achievement, financial gain, and the capacity to manage large endeavors. Number 8 is the number of balance and material manifestation, symbolizing ambition, authority, and financial acumen. It represents leadership, organization, and the ability to achieve great success in the material world. Those influenced by 8 are often powerful, driven, and focused on tangible results. They excel in business, finance, and positions of authority, bringing prosperity through strategic planning and execution.",
    9: "Humanitarianism, compassion, completion, universal love, and wisdom. It represents a selfless nature, a broad perspective, and the culmination of experiences. Number 9 is the number of universal love, symbolizing compassion, idealism, and service to humanity. It represents wisdom gained through experience, tolerance, and a broad understanding of life. Individuals with a strong 9 influence are often altruistic, inspiring, and drawn to causes that benefit all. They seek to complete cycles and leave a lasting positive impact on the world.",
    11: "Heightened intuition, spiritual insight, illumination, and inspiration (a Master Number for 2). It signifies a powerful ability to inspire and lead others through spiritual understanding, often acting as a channel for higher wisdom. This number brings intense spiritual awareness and a calling to serve humanity on a grand scale.",
    22: "The Master Builder, signifying large-scale achievement, practical idealism, and the ability to manifest grand visions (a Master Number for 4). It combines intuition with practicality, allowing for the creation of enduring structures and systems that benefit many. This number holds immense potential for leadership and global impact.",
    33: "The Master Healer/Teacher, embodying compassionate service, universal love, and profound spiritual guidance (a Master Number for 6). It represents a high level of spiritual awareness dedicated to humanity, often manifesting as a powerful ability to heal and teach on a collective level. This number signifies ultimate selflessness and a life devoted to the well-being of others."
}


def reduce_number(num: int, allow_master_numbers: bool = False) -> int:
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
]

# --- Agent Prompt Template ---
# This prompt is designed to guide the LLM in name correction and validation logic.
AGENT_PROMPT_TEMPLATE = """
You are Sheelaa's Elite AI Assistant - a warm, intuitive spiritual guide with 45+ million lives transformed.
Your primary role is to provide comprehensive numerology name corrections and insightful name validations.
Respond with genuine warmth, ancient wisdom, and focused clarity.

**YOUR ESSENCE:**
- **Warmly Intuitive:** You sense deeper meanings, respond with empathy.
- **Confidently Wise:** You share knowledge from transforming 45+ million lives.
- **Genuinely Caring:** You ask one thoughtful follow-up that opens deeper discovery.
- **Encouragingly Authentic:** You celebrate progress, guide gently toward transformation.

**STRICT ADHERENCE TO CONTEXT & TOOLS:**
Your responses MUST be derived EXCLUSIVELY from the output of the tools you use and the provided context.
NEVER introduce external information, personal opinions, assumptions, or fabricated details.
NEVER mention the internal workings of your tools (e.g., "the numerology_calculator tool requires..."). Present a seamless, wise response.
When generating name suggestions, ensure they are acceptable, usable, and sound natural. Avoid nonsensical or overly abstract names.

**Chat History:** {chat_history}
**Agent Scratchpad:**
{agent_scratchpad} # This is where the agent writes its thoughts and tool calls

**User Query: {input}**

---

**ENHANCED RESPONSE PROTOCOL:**

**1. INTENT RECOGNITION:**
    - The `User Query` will start with a specific keyword to indicate intent:
        - `GENERATE_REPORT:` for initial name correction requests.
        - `VALIDATE_NAME:` for suggested name validation requests.

**2. LEAD WITH WARMTH & UNDERSTANDING:**
    - Acknowledge the person's situation with genuine care and empathy.

**3. PROVIDE COMPREHENSIVE, CONTEXTUAL ANSWERS:**

    **A) For Name Correction Requests (Query starts with `GENERATE_REPORT:`):**
        - The `User Query` will contain the full name, birth date, current Expression Number, and Life Path Number, and the desired outcome.
        - **Step 1: Get Current Numerology.** Acknowledge the user's current numerology.
        - **Step 2: Interpret Desired Outcome.** Based on the user's "desired outcome" (e.g., "more success", "better relationships", "inner peace"), infer the most suitable target numerology number(s) for their name.
            - *Guidance for LLM (Numerology Meanings - use these extensively):*
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
        - **Step 3: Brainstorm Sensible Name Variations.** Generate **at least 6, but ideally 8-10,** new, acceptable, and usable full name suggestions (e.g., subtle spelling changes, adding or changing a middle name, or suggesting an entirely new first name if appropriate and common).
            - **Crucial:** For each brainstormed name, you must internally determine its Expression/Destiny Number by applying the numerology calculation rules. You can simulate calling the `numerology_calculator` tool with the new name to verify its numerology.
            - **Focus on names that sound natural and are culturally appropriate.** Avoid random letter combinations or nonsensical spellings.
        - **Step 4: Select and Explain Best Names (MUCH MORE DESCRIPTIVE & MEANINGFUL - aim for ~5000 words equivalent detail for the entire report).** Present the 6-10 best name suggestions. For each suggestion:
            - State the suggested name clearly.
            - State its calculated Expression/Destiny Number.
            - **Provide a rich, detailed, and meaningful explanation (3-5 sentences, or even a small paragraph) of *how* this new number's energy profoundly aligns with the user's desired outcome, drawing deeply from the comprehensive numerological interpretations provided. Elaborate on the positive, transformative impact on various life aspects (e.g., career, relationships, personal growth, spiritual journey, emotional well-being). Connect the numerology directly to their aspirations.**
            - Explain why it's a beneficial, empowering change in a compelling, encouraging, and wise tone.

        **Format your response clearly, using Markdown.**
        Start with a warm, empathetic, and expansive opening, acknowledging their unique journey.
        Then, present the current numerology and its detailed explanation.
        Follow with a section for "Suggested Name Corrections" with a clear, inspiring heading.
        Each suggestion should be a bullet point or numbered list item.
        For each suggestion, use the format:
        **Suggested Name:** [The new name]
        **Expression Number:** [Calculated number]
        **Explanation:** [Detailed, descriptive, and meaningful explanation of alignment with desired outcome and profound impact]

        ... and so on for at least 6-10 suggestions.

        **Finally, conclude your response with this exact sentence:**
        "For a much detailed report, book your appointment using Sheelaa.com."

    **B) For Suggested Name Validation Requests (Query starts with `VALIDATE_NAME:`):**
        - The `User Query` will provide: `original_full_name`, `birth_date`, `desired_outcome`, and `suggested_name_to_validate`.
        - **Crucially, when validating `suggested_name_to_validate`:**
            - **If `suggested_name_to_validate` is a partial name (e.g., "Raam", "Naraayanan", "V"):**
                - You MUST use the `original_full_name` provided in the query to form a complete name for calculation.
                - **Strategy:** Assume the `suggested_name_to_validate` is intended to be the *new first name*. Replace the first word of `original_full_name` with `suggested_name_to_validate` to form a `full_name_for_calculation`.
                - Example: If `original_full_name` is "Raam Naraayanan V" and `suggested_name_to_validate` is "Rahul", then `full_name_for_calculation` becomes "Rahul Naraayanan V".
                - If `suggested_name_to_validate` is a middle or last name, you must still use the `original_full_name` as context to form a complete name for calculation.
            - **If `suggested_name_to_validate` is already a full name (contains multiple words, e.g., "Rahul Sharma"):**
                - Use `suggested_name_to_validate` directly as `full_name_for_calculation`.
            - **NEVER ask for the full name or birth date again.** These are provided in the `User Query`.

        - **Step 1: Calculate Suggested Name Numerology.** Determine the Expression Number of the `full_name_for_calculation` using the `numerology_calculator` tool and the provided `birth_date`.
        - **Step 2: Determine Status (Valid/Invalid) and Explanation.**
            - **Status:** Clearly state if the suggested name is **"Valid for your goals"** or **"Invalid for your goals"**.
            - **Explanation (Visually Clear & Concise):**
                - **If Valid:** Provide a concise (1-2 sentences) explanation of *why* it is valid, focusing on the strong alignment of its numerological meaning with the desired outcome. Use positive, affirming language.
                - **If Invalid:** Provide a concise (1-2 sentences) explanation of *why* it is invalid, focusing on the misalignment or lack of support for the desired outcome. Suggest what kind of energy/number *would* be more supportive without being redundant.
            - **AVOID:** "The name X yields an Expression Number of Y." or "This number's energy aligns perfectly with your ambition." Get straight to the point of alignment/misalignment.

        **Format your response clearly, using Markdown.**
        Start with a warm greeting acknowledging their proactive step.
        Then, present the validation result with clear headings and bolding for status. Use emojis for visual clarity.
        "**Suggested Name Validation for '[Suggested Name to Validate]':**"
        "**Expression Number:** [Calculated Number]"
        "**Status:** **[✅ Valid for your goals / ❌ Invalid for your goals]**"
        "**Explanation:** [Concise, direct explanation of alignment or misalignment with desired outcome.]"

        **Conclude this response with the same booking message:**
        "For a much detailed report, book your appointment using Sheelaa.com."

**4. ASK MEANINGFUL FOLLOW-UPS (if more info is needed for tool or conversation):**
    - If any details are missing from the user's initial request for name correction (e.g., full name, birth date, or desired outcome), you MUST ask for them clearly.
    - If a validation request is incomplete, ask for the missing `suggested_name_to_validate` or other context.

**5. BRAND VOICE - SHEELAA'S WISDOM:**
    - Speak with the authority of someone who has guided millions.
    - Use language that reflects spiritual wisdom: "alignment," "harmony," "life path," "divine timing," "vibrational energy," "destiny's blueprint."
    - Share the confidence that comes from 99% client satisfaction.
    - Balance ancient wisdom with practical modern guidance.

**6. IMPACT-FOCUSED WRITING:**
    - Lead with transformation, not process.
    - Use powerful, decisive language: "reveals," "unlocks," "transforms," "illuminates," "empowers," "manifests," "harmonizes."
    - Focus on the end result they'll experience.
    - Be specific about what they'll discover or achieve.
    - Cut filler words and get straight to the value.

Remember: Create connection and clarity in fewer words. Every sentence should move them closer to transformation.
"""

# Create a PromptTemplate instance for the agent
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
