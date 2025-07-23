from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import datetime
import re
import json

# Langchain imports (mostly for ChatGoogleGenerativeAI, others are less relevant now)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables from .env file (for local testing)
load_dotenv()

app = Flask(__name__)

# --- CORS Configuration ---
CORS(app, resources={r"/*": {"origins": [
    "https://namecorrectionsheelaa.netlify.app", # Your Netlify frontend domain
    "http://localhost:3000" # For local development
]}})


# --- Global Variables for LLM and Memory ---
llm = None # This LLM will be used for all direct calls
memory = None # Still useful for maintaining conversation history

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Numerology Calculator Core Logic ---
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

# Comprehensive Interpretations for the LLM to use
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

# This function is now a direct Python utility
def get_numerology_details(full_name: str, birth_date: str) -> dict:
    """
    Calculates Expression/Destiny and Life Path Numbers.
    Returns a dictionary with details or raises ValueError.
    """
    if not full_name or not birth_date:
        raise ValueError("Both full name and birth date are required.")
    
    expression_number = calculate_name_number(full_name)
    life_path_number = calculate_life_path_number(birth_date)
    
    return {
        "full_name": full_name,
        "birth_date": birth_date,
        "expression_number": expression_number,
        "life_path_number": life_path_number
    }


# --- Function to Initialize LLM ---
def initialize_llm():
    global llm, memory

    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        return False

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7, # Balanced for creativity and factual interpretation
            top_p=0.9,
            top_k=40
        )
        print("LLM (Gemini) initialized successfully.")

        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
        print("Conversational memory initialized.")

        return True

    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return False

# Initialize LLM on app startup
if not initialize_llm():
    print("Failed to initialize LLM on startup. API will not function correctly.")

# --- Chat Endpoint ---
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    print(f"Received message: {user_message}")

    if llm is None:
        print("Error: LLM not initialized.")
        return jsonify({"error": "Chatbot not fully initialized. Please check backend logs."}), 500

    try:
        if user_message.startswith("GENERATE_REPORT:"):
            # --- Handle Report Generation ---
            # Parse initial user details
            report_data_match = re.search(r"My full name is \"(.*?)\" and my birth date is \"(.*?)\"\. My current Name \(Expression\) Number is (\d+) and Life Path Number is (\d+)\. I desire the following positive outcome in my life: \"(.*?)\"\.", user_message)
            
            if not report_data_match:
                return jsonify({"error": "Invalid format for GENERATE_REPORT message."}), 400
            
            original_full_name, birth_date, current_exp_num_str, current_life_path_num_str, desired_outcome = report_data_match.groups()
            current_exp_num = int(current_exp_num_str)
            current_life_path_num = int(current_life_path_num_str)

            # --- Step 1: Generate Initial Report Introduction ---
            intro_prompt = f"""
            You are Sheelaa's Elite AI Assistant.
            Given the user's details:
            Full Name: "{original_full_name}"
            Birth Date: "{birth_date}"
            Current Expression Number: {current_exp_num}
            Current Life Path Number: {current_life_path_num}
            Desired Outcome: "{desired_outcome}"

            Write a warm, empathetic, and expansive opening for their personalized numerology report.
            Acknowledge their unique journey and current numerological profile.
            Explain the combined energy of their current Expression ({current_exp_num}) and Life Path ({current_life_path_num}) numbers, drawing from the following comprehensive interpretations:
            {json.dumps(NUMEROLOGY_INTERPRETATIONS, indent=2)}
            Conclude this introduction by setting the stage for name corrections to amplify their aspirations.
            Ensure the tone is professional, wise, and deeply caring. Avoid any mention of tools or calculation processes.
            Aim for a detailed introduction, equivalent to several paragraphs.
            """
            intro_response = llm.invoke(intro_prompt).content
            full_report_content = intro_response + "\n\n"

            full_report_content += "## ✨ Suggested Name Corrections:\n\n"

            # --- Step 2: Brainstorm Names (LLM structured output) ---
            # Request 8-10 names from LLM, then calculate their numbers in Python
            name_brainstorm_prompt = f"""
            You are Sheelaa's Elite AI Assistant.
            Given the user's details:
            Original Full Name: "{original_full_name}"
            Birth Date: "{birth_date}"
            Desired Outcome: "{desired_outcome}"
            Numerology Interpretations: {json.dumps(NUMEROLOGY_INTERPRETATIONS, indent=2)}

            Your task is to brainstorm 8-10 sensible, usable, and culturally appropriate full name suggestions.
            These should be variations of the original name or new first/middle names that sound natural.
            Output your suggestions as a JSON array of strings (full names), like this:
            ```json
            [
              "New Name One",
              "New Name Two",
              "New Name Three"
            ]
            ```
            Ensure the JSON is valid and complete.
            """
            
            llm_for_names = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.9, # Higher temperature for more creative names
                top_p=0.9,
                top_k=40
            )

            name_suggestions_raw = llm_for_names.invoke(name_brainstorm_prompt).content
            
            # Extract JSON from potential markdown code block
            name_suggestions_json_match = re.search(r"```json\n(.*?)```", name_suggestions_raw, re.DOTALL)
            if name_suggestions_json_match:
                suggested_names_list = json.loads(name_suggestions_json_match.group(1))
            else:
                # Fallback if LLM doesn't wrap in ```json
                suggested_names_list = json.loads(name_suggestions_raw) 

            # --- Step 3: Calculate Actual Expression Numbers and Generate Explanations ---
            for suggested_name_full in suggested_names_list:
                try:
                    calculated_details = get_numerology_details(suggested_name_full, birth_date)
                    actual_expression_number = calculated_details["expression_number"]
                except ValueError as e:
                    print(f"Error calculating numerology for {suggested_name_full}: {e}")
                    actual_expression_number = "N/A" # Should not happen with correct input

                explanation_prompt = f"""
                You are Sheelaa's Elite AI Assistant.
                User's Original Full Name: "{original_full_name}"
                User's Birth Date: "{birth_date}"
                User's Desired Outcome: "{desired_outcome}"
                Suggested Name: "{suggested_name_full}"
                Calculated Expression Number for Suggested Name: {actual_expression_number}
                Comprehensive Numerology Interpretations: {json.dumps(NUMEROLOGY_INTERPRETATIONS, indent=2)}

                Write a rich, detailed, and meaningful explanation (aim for a paragraph of 3-5 sentences) for the suggested name.
                Explain *how* its calculated Expression Number ({actual_expression_number}) profoundly aligns with the user's desired outcome ("{desired_outcome}").
                Elaborate on the positive, transformative impact on various life aspects (e.g., career, relationships, personal growth, spiritual journey, emotional well-being) by drawing deeply from the provided interpretations.
                Explain why it's a beneficial, empowering change in a compelling, encouraging, and wise tone.
                Ensure the explanation is unique and does not repeat information from other suggestions.
                Avoid any mention of tools or calculation processes.
                """
                
                explanation_response = llm.invoke(explanation_prompt).content
                
                full_report_content += f"""
**Suggested Name:** {suggested_name_full}
**Expression Number:** {actual_expression_number}
**Explanation:** {explanation_response}

"""
            # Final conclusion for the report
            full_report_content += "For a much detailed report, book your appointment using Sheelaa.com."


            bot_response = full_report_content

        elif user_message.startswith("VALIDATE_NAME:"):
            # --- Handle Name Validation ---
            # Robust parsing of the VALIDATE_NAME message
            validation_data_match = re.search(r"Original Full Name: \"(.*?)\", Birth Date: \"(.*?)\", Desired Outcome: \"(.*?)\", Suggested Name to Validate: \"(.*?)\"", user_message)
            
            if not validation_data_match:
                return jsonify({"error": "Invalid format for VALIDATE_NAME message."}), 400

            original_full_name, birth_date, desired_outcome, suggested_name_to_validate = validation_data_match.groups()

            # Logic to determine the full name for calculation based on suggested_name_to_validate
            full_name_for_calculation = suggested_name_to_validate.strip()
            original_name_parts = original_full_name.split()

            # If suggested name is a single word and original name has multiple parts, assume it's a new first name
            if len(full_name_for_calculation.split()) == 1 and len(original_name_parts) > 1:
                # Replace the first name with the suggested name, preserving middle/last names
                full_name_for_calculation = full_name_for_calculation + " " + " ".join(original_name_parts[1:])
            elif len(full_name_for_calculation.split()) == 1 and len(original_name_parts) == 1:
                # If original name was single word and suggested is single word, just use suggested
                pass # full_name_for_calculation is already suggested_name_to_validate
            # If suggested_name_to_validate has multiple words, assume it's a full name already

            print(f"Validating: '{suggested_name_to_validate}' in context of original full name: '{original_full_name}'")
            print(f"Calculated name for tool: '{full_name_for_calculation}'")

            # Directly call the numerology calculator function
            try:
                calculated_details = get_numerology_details(full_name_for_calculation, birth_date)
                calculated_expression_number = calculated_details["expression_number"]
            except ValueError as e:
                print(f"Error calculating numerology for validation: {e}")
                return jsonify({"error": f"Could not calculate numerology for validation: {e}"}), 500

            # Now, pass this pre-calculated result to the LLM for interpretation and formatting
            validation_interpretation_prompt = f"""
            You are Sheelaa's Elite AI Assistant.
            Original Full Name: "{original_full_name}"
            Birth Date: "{birth_date}"
            Desired Outcome: "{desired_outcome}"
            Suggested Name to Validate: "{suggested_name_to_validate}"
            Calculated Expression Number for '{full_name_for_calculation}': {calculated_expression_number}
            Comprehensive Numerology Interpretations: {json.dumps(NUMEROLOGY_INTERPRETATIONS, indent=2)}

            Your task is ONLY to interpret the provided `calculated_expression_number` for `suggested_name_to_validate` against the `desired_outcome`.
            NEVER ask for the full name or birth date again. These are provided in the context.
            NEVER mention internal tools or that a calculation was performed by a tool.
            
            Determine if the suggested name is "Valid for your goals" or "Invalid for your goals".
            
            **Format your response clearly, using Markdown, with bolding for status and emojis:**
            "**Suggested Name Validation for '[Suggested Name to Validate]':**"
            "**Expression Number:** {calculated_expression_number}"
            "**Status:** **[✅ Valid for your goals / ❌ Invalid for your goals]**"
            "**Explanation:** [Concise, direct explanation of alignment or misalignment with desired outcome. If invalid, suggest what kind of energy/number would be more supportive without being redundant.]"
            Conclude with: "For a much detailed report, book your appointment using Sheelaa.com."
            """
            
            llm_response = llm.invoke(validation_interpretation_prompt)
            bot_response = llm_response.content

        else:
            # Fallback for unexpected messages
            bot_response = "I can either generate a personalized numerology report or validate a suggested name. Please start your query with 'GENERATE_REPORT:' or 'VALIDATE_NAME:' followed by your details."

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
