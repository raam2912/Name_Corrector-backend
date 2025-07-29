import os
import datetime
import re
import json
import asyncio
import logging
from functools import wraps
from typing import Counter, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
import time
import secrets
import hmac
import sys
import random

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from dotenv import load_dotenv

# NEW: Import WSGIMiddleware for Flask (WSGI) to Uvicorn (ASGI) compatibility
from uvicorn.middleware.wsgi import WSGIMiddleware

# ReportLab specific imports for PDF generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.colors import HexColor
import io

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('numerology_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app

# === PATCH: Chaldean Strict Name Validation ===
LUCKY_NAME_NUMBERS = {1, 3, 5, 6, 9, 11, 22, 33}
UNLUCKY_NAME_NUMBERS = {4, 8}
KARMIC_DEBT_NUMBERS = {13, 14, 16, 19}

EXPRESSION_COMPATIBILITY_MAP = {
    1: {1, 5, 6, 9},
    2: {2, 4, 6},
    3: {3, 6, 9},
    4: {1, 5, 6},
    5: {1, 5, 6},
    6: {3, 6, 9},
    7: {2, 7},
    8: {1, 5, 6},
    9: {3, 6, 9},
    11: {2, 6, 11},
    22: {4, 8, 22},
    33: {3, 6, 9, 33},
}
CHALDEAN_MAP = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 8, 'G': 3,
    'H': 5, 'I': 1, 'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5,
    'O': 7, 'P': 8, 'Q': 1, 'R': 2, 'S': 3, 'T': 4, 'U': 6,
    'V': 6, 'W': 6, 'X': 5, 'Y': 1, 'Z': 7
}

MASTER_NUMBERS = {11, 22, 33}
KARMIC_DEBT_NUMBERS = {13, 14, 16, 19}

EXPRESSION_COMPATIBILITY_MAP = {
    1: [1, 3, 5, 6],
    2: [2, 4, 6, 9],
    3: [1, 3, 5, 6, 9],
    4: [1, 5, 6],
    5: [1, 3, 5, 6, 9],
    6: [3, 5, 6, 9],
    7: [1, 5, 6, 9],
    8: [1, 3, 5, 6],
    9: [3, 6, 9],
    11: [2, 6, 11, 22],
    22: [4, 6, 8, 22],
    33: [6, 9, 33]
}

def calculate_expression_number(name):
    total = sum(CHALDEAN_MAP.get(c.upper(), 0) for c in name if c.isalpha())
    return reduce_to_single_or_master(total), total

def is_lucky_number(num):
    return num in {1, 3, 5, 6, 9, 11, 22, 33}

def is_karmic_debt_number(num):
    return num in KARMIC_DEBT_NUMBERS

def reduce_to_single_or_master(n):
    while n > 9 and n not in MASTER_NUMBERS:
        n = sum(int(d) for d in str(n))
    return n
def is_expression_number_strictly_valid(expr_num: int, life_path: int, birth_number: int) -> bool:
    if expr_num not in LUCKY_NAME_NUMBERS:
        return False
    if expr_num in UNLUCKY_NAME_NUMBERS:
        return False
    if life_path not in EXPRESSION_COMPATIBILITY_MAP.get(expr_num, set()):
        return False
    if birth_number not in EXPRESSION_COMPATIBILITY_MAP.get(expr_num, set()):
        return False
    return True
# === END PATCH ===

app = Flask(__name__)
logger.info("Flask app instance created.")
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'numerology-secret-key-2024')
app.config['CACHE_TYPE'] = 'simple'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300

# Initialize extensions (cache and limiter)
cache = None
limiter = None
try:
    cache = Cache(app)
    limiter = Limiter(
        app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri=os.getenv('REDIS_URL', 'memory://')
    )
    logger.info("Flask-Caching and Flask-Limiter initialized successfully.")
except Exception as e:
    logger.warning(f"Could not initialize Redis features (Caching/Limiter might be unavailable): {e}")

# CORS Configuration
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

logger.info("CORS configured for the Flask app.")

# --- Decorators ---
def cached_operation(timeout=3600): # Increased cache timeout for profile data
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if cache is None:
                logger.warning(f"Cache not initialized, skipping caching for {func.__name__}.")
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)

            # Create a unique cache key based on function name and arguments
            key_parts = [func.__name__] + [str(arg) for arg in args]
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
            cache_key = hashlib.md5("_".join(key_parts).encode()).hexdigest()

            cached_result = cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for {func.__name__}")
                return cached_result
            
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            cache.set(cache_key, result, timeout=timeout)
            logger.info(f"Cache miss for {func.__name__}, result cached.")
            return result
        return wrapper
    return decorator

# --- Security Manager ---
class SecurityManager:
    @staticmethod
    def validate_input_security(input_string: str) -> bool:
        """
        A placeholder for input security validation.
        Checks for common script tags or SQL keywords.
        """
        if re.search(r'<(script|iframe|img|link|style).*?>', input_string, re.IGNORECASE) or \
           re.search(r'(SELECT|INSERT|UPDATE|DELETE|DROP)\s+', input_string, re.IGNORECASE):
            return False
        return True

# --- Pydantic Schemas for Output Parsing ---
class NameSuggestion(BaseModel):
    name: str = Field(description="The suggested full name variation.")
    rationale: str = Field(description="Detailed explanation for the suggested name, including numerological advantages, connection to desired outcome, and energetic transformation.")
    expression_number: int = Field(description="The calculated Expression Number for the suggested name.")

class NameSuggestionsOutput(BaseModel):
    suggestions: List[NameSuggestion] = Field(description="A list of suggested name variations.")
    reasoning: str = Field(description="Overall reasoning for the name suggestion strategy.")

parser = PydanticOutputParser(pydantic_object=NameSuggestionsOutput)

# --- LLM Manager ---
class LLMManager:
    def __init__(self):
        self.llm = None
        self.creative_llm = None
        self.analytical_llm = None
        self.memory = None
        self._initialize_llms()

    def _initialize_llms(self):
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if not google_api_key:
            logger.error("GOOGLE_API_KEY environment variable not set.")
            raise ValueError("GOOGLE_API_KEY is not set. Please set it to use the Generative AI models.")

        try:
            # Using gemini-1.5-flash as requested
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0.7, convert_system_message_to_human=True)
            self.creative_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0.9, convert_system_message_to_human=True)
            self.analytical_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0.3, convert_system_message_to_human=True)
            
            self.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
            
            logger.info("All LLM instances initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM instances: {e}")
            raise

llm_manager = LLMManager()
logger.info("Initializing Numerology Application...")
try:
    llm_manager._initialize_llms()
    logger.info("Application initialized successfully")
except Exception as e:
    logger.error(f"Application initialization failed: {e}")
    sys.exit(1)

# --- REFINED PROMPTS FOR NUMEROLOGY APPLICATION ---
# Removed any mention of "desired outcome" from prompts
NAME_SUGGESTION_SYSTEM_PROMPT = """You are Sheelaa's Elite AI Numerology Assistant and Master Name Strategist. Your expertise lies in creating numerologically aligned name variations that preserve cultural authenticity while optimizing energetic outcomes.

## YOUR MISSION:
Generate **12** strategically crafted full name variations that:
- Maintain maximum similarity to the original name (90%+ resemblance)
- Align precisely with optimal Expression Numbers (chosen for general well-being and balance)
- Sound natural, culturally appropriate, and **highly practical for real-world use**
- **CRITICAL PRIORITY: Numerological validity (correct Expression Number, avoidance of Karmic Debts, alignment with core numbers) takes precedence over phonetic harmony if a trade-off is necessary.**

## MODIFICATION GUIDELINES:
- **Minimal Changes Only**: Single letter alterations, spelling variations, middle initial additions/removals
- **Preserve Core Identity**: Keep the essence and pronunciation as close as possible
- **Cultural Sensitivity**: Ensure variations respect the original name's cultural context
- **Practicality First**: Prioritize names that are easy to adopt and integrate into daily life. Avoid overly complex or unusual suggestions.

## RATIONALE REQUIREMENTS:
For each suggestion, provide a comprehensive 2-3 sentence explanation that:
1. **Specifies the exact numerological advantage** of the new Expression Number
2. **Explains the energetic transformation** this change creates
3. **Emphasizes positive impact** and specific benefits

**When crafting rationales, consider these additional factors from the client's profile (if available):**
- **Lo Shu Grid Balance**: How the new name helps balance missing energies (e.g., if 5 is missing, does the new name's Expression 5 help?)
- **Planetary Compatibility**: How the new name's planetary ruler (from Chaldean mapping) aligns with their Ascendant/Lagna or avoids conflicts with malefic planetary lords.
- **Phonetic Harmony**: Confirm the suggested name maintains a pleasant and strong vibrational tone.
- **Edge Case Resolutions**: If the original profile had an exact edge case, explain how the new name resolves or mitigates it.

## OUTPUT FORMAT:
Return a valid JSON object conforming to NameSuggestionsOutput schema with accurate expression_number calculations.
**CRITICAL: DO NOT include any comments (e.g., // comments) in the JSON output.**

## QUALITY STANDARDS:
- Each rationale must be substantive, specific, and compelling
- Avoid generic explanations - make each one unique and targeted
- If a target number cannot be achieved while maintaining similarity, acknowledge this in reasoning""" + "{parser_instructions}"

NAME_SUGGESTION_HUMAN_PROMPT = """**NUMEROLOGICAL NAME OPTIMIZATION REQUEST**
**Original Name:** "{original_full_name}"
**TASK:** Create 12 name variations that are nearly identical to the original but numerologically optimized for general well-being and balance. Each suggestion must include a detailed, specific rationale explaining its advantages.

**REQUIREMENTS:**
- Maintain 90%+ similarity to original name
- Aim for a diverse set of optimal Expression Numbers for general positive influence.
- Provide compelling, detailed explanations for each suggestion
- Ensure cultural appropriateness, natural sound, and **high practicality for adoption**"""

ADVANCED_REPORT_SYSTEM_PROMPT = """You are Sheelaa's Elite AI Numerology Assistant, renowned for delivering transformative, deeply personalized numerological insights. You create comprehensive reports that combine ancient wisdom with modern psychological understanding, **integrating Chaldean Numerology, Lo Shu Grid analysis, Astro-Numerology principles (based on provided data), and Phonology considerations.**

## REPORT STRUCTURE (Follow Exactly):
### 1. **Executive Summary** (Concise & Impactful)
- Provide a high-level overview of the client's core numerological blueprint.
- Summarize the primary insights and the most impactful name correction.
- Highlight the key benefits of adopting the new name for overall well-being and energetic alignment.
- This section should be concise but powerful, setting the stage for the detailed report.

### 2. **Introduction** (Warm & Personal)
- Acknowledge their name and birth date personally, reinforcing the sacred journey.
- Create immediate connection and establish trust, emphasizing the report's transformative potential.
- Set a positive, empowering tone for the entire report.
- Explain the holistic approach, integrating various numerological and conceptual astrological systems.

### 3. **Core Numerological Blueprint: A Deep Dive** (Extensive Analysis)
- **Expression Number Deep Dive**:
    - Provide a multi-paragraph analysis of its core traits, psychological patterns, natural abilities, and its Chaldean planetary association.
    - Discuss how this number shapes their public persona and life's work.
    - Include specific examples of how this energy might manifest in daily life and career.
- **Life Path Number Analysis**:
    - Offer a multi-paragraph exploration of their life purpose, karmic direction, and soul mission.
    - Detail the overarching themes and lessons they are destined to encounter.
    - Connect this to their inherent strengths and challenges.
- **Birth Day Number Significance**:
    - Explain in detail the day of birth's influence on their personality and immediate characteristics.
    - Discuss how it complements or contrasts with their Expression and Life Path numbers.
- **Compatibility Synthesis**:
    - Provide a thorough analysis of how these core numbers (Expression, Life Path, Birth Day) work together or create dynamic challenges.
    - Offer insights into areas of natural synergy and areas requiring conscious integration.
    - Draw from provided interpretations but add unique, personalized insights.

### 4. **Soul Architecture: Inner & Outer Selves** (Detailed Exploration)
- **Soul Urge Number**:
    - Delve into their inner motivations, heart's desires, and what truly fulfills them on a soul level.
    - Explain how this internal drive influences their choices and aspirations.
- **Personality Number**:
    - Describe in detail how others perceive them, their outer persona, and the first impressions they make.
    - Discuss how this external presentation can be leveraged for their overall well-being.
- **🌟 STRATEGIC NAME CORRECTIONS: Your Path to Transformation** (CRITICAL, Highly Detailed)
- Present *only the confirmed suggested names provided in the input*, each with its Expression Number.
- **Use the exact detailed rationales provided in the input data.**
- Explain in multiple paragraphs how each suggestion specifically enhances their overall well-being and energetic alignment, providing concrete scenarios.
- Compare energetic shifts from current to suggested numbers with rich descriptive language.
- Provide comprehensive implementation guidance, including psychological preparation and practical steps for adopting the new name.
- Offer detailed timing recommendations for name adoption, linking to their personal year cycles.
- **Crucially, discuss in depth how each suggested name addresses specific numerological or astrological imbalances identified in the client's profile (e.g., balancing missing Lo Shu numbers, harmonizing with planetary influences, mitigating karmic debts). Provide specific examples related to their profile.**

### 6. **Karmic Lessons & Soul Work: Unlocking Growth** (In-depth Analysis)
- Detail **all** missing numbers from their **Lo Shu Grid** and their profound significance, providing multiple paragraphs of interpretation for each.
- Explain how consciously addressing these lessons accelerates personal and spiritual growth.
- **Specifically mention any karmic debt numbers identified in their birth date or original name, explaining their origin and the specific lessons they present.**
- Offer practical exercises or mindset shifts for integrating these lessons.

### 7. **Temporal Numerology: Navigating Your Cycles** (Comprehensive Timing)
- Provide an extensive analysis of their Current Personal Year energy, including detailed optimal activities and potential challenges.
- Explain the significance of the best months for implementing changes (e.g., name changes, new ventures), providing the numerological reasoning behind these periods.
- Offer a broader energy forecast for the current and upcoming periods, guiding strategic planning.

### 8. **Shadow Work & Growth Edge: Embracing Wholeness** (Honest & Empowering Assessment)
- Detail potential challenges specific to their numbers, **including all identified edge cases (e.g., Expression 8/Life Path 1 conflict, Master Number challenges), providing in-depth explanations of their manifestations.**
- Offer multiple, practical, and actionable mitigation strategies for each identified challenge.
- Reframe challenges as profound growth opportunities, guiding them towards transformation.

### 9. **Personalized Development Blueprint: Actionable Steps** (Highly Practical)
- Outline immediate focus areas with specific, actionable steps for daily integration.
- Detail a long-term vision aligned with their numbers, providing a roadmap.
- Suggest comprehensive monthly practices for sustained growth and energetic alignment.
- **Provide specific actions that help integrate energies from their Lo Shu Grid or align with beneficial planetary influences, offering concrete examples.**

### 10. **Future Cycles Forecast: Your 3-Year Vision** (Detailed Predictive Insights)
- Provide a year-by-year energy theme and focus areas for the next three years, with detailed interpretations for each year.
- Offer strategic planning advice for optimal outcomes in each year.
- Suggest a timeline for major life decisions within these cycles.

### 11. **Numerical Uniqueness Profile: Your Cosmic Signature** (Special Recognition)
- Highlight all rare combinations or master numbers in their profile, explaining their unique gifts and cosmic significance in multiple paragraphs.
- **Discuss in detail how their unique numerical blueprint interacts with their conceptual astrological parameters (Ascendant/Lagna, Moon Sign, Planetary Lords), providing a rich tapestry of interwoven influences.**

### 12. **Empowerment Conclusion** (Inspiring & Action-Oriented Close)
- Synthesize all key insights from the report, reinforcing their inherent power.
- Reiterate their capacity to shape destiny through conscious alignment.
- Leave them feeling profoundly inspired, capable, and equipped with actionable wisdom.
IMPORTANT REQUIREMENT:
All suggested names MUST already be strictly numerologically valid:
- The name must align with the person's Life Path Number and Birth Day Number make it so its advantageous and follows the lucky name correction of Chaldean numerology.
- It must have a valid Expression Number (as per Chaldean numerology rules).
- Avoid any name combinations that would result in karmic debts.
- Do NOT suggest names unless they fully pass strict numerological evaluation.

Return ONLY names that fully meet the above rules. No post-filtering will be applied.
## WRITING STANDARDS:
- **Depth & Breadth**: Generate *extensive* content for each section and sub-section. **Aim for a minimum of 50 pages for the complete report, ensuring each point is elaborated with rich examples, detailed scenarios, and profound insights. Maximize descriptive language and provide comprehensive analysis.**
- **Professional & Authoritative Tone**: Maintain the voice of a master numerologist – wise, insightful, and empowering. Avoid casual language. The tone should be academic, deeply analytical, and highly credible.
- **Personal Touch**: Use their name throughout, make it feel custom-crafted and deeply personal.
- **Markdown Formatting**: Use headers (`#`, `##`, `###`), bold text (`**text**`), italic text (`*text*`), and bullet points (`* item`).
- **Highlighting for Readability**:
    - **Crucial Takeaway**: For key insights or critical advice, start a paragraph with "<b>Crucial Takeaway:</b> " followed by the insight. Use this frequently for vital information.
    - **Key Insight**: For significant revelations, start a paragraph with "<b>Key Insight:</b> " followed by the revelation.
    - **Important Note**: For cautionary or vital information, start a paragraph with "<b>Important Note:</b> " followed by the note.
    - Use `**bold**` for all numerological terms (e.g., **Expression Number**, **Life Path Number**) and key concepts.
- **STRICT ADHERENCE TO NAMES**: When referring to the client's original or suggested names, you MUST use the exact spelling provided in the input data. Do NOT alter or simplify the spelling of names (e.g., if the input is 'Naraayanan', use 'Naraayanan', not 'Narayanan').
- **Specific Details**: Avoid generalities. Provide concrete insights, **referencing specific numbers, grid positions, planetary influences, and real-world implications where relevant.**
- **Cohesion & Flow**: Despite the extensive detail, ensure the content flows logically and is highly coherent, guiding the reader through their numerological journey seamlessly.
- **No AI Disclaimers.**
- **Assume the reader is a practitioner who appreciates deep, academic-level numerological analysis.**
"""

ADVANCED_REPORT_HUMAN_PROMPT = """**COMPREHENSIVE NUMEROLOGY REPORT REQUEST**
Please generate a detailed, transformational numerology report using this complete profile data:
```json
{llm_input_data}
```
**REQUIREMENTS:**
- Follow the exact 12-section structure outlined in your instructions.
- Elaborate extensively on each section and sub-section - aim for truly comprehensive analysis, providing multiple paragraphs and examples for each point.
- The 'Strategic Name Corrections' section should ONLY use the 'confirmed_suggestions' provided in the JSON input, along with their rationales.
- Create a report that feels personally crafted, professionally valuable, and profoundly insightful."""

# Removed GENERAL_CHAT_SYSTEM_PROMPT and NAME_VALIDATION_SYSTEM_PROMPT as chat functionality is removed.

# --- Helper Functions (Numerology Calculations) ---
# These functions are globally defined and accessible by all endpoints.

CHALDEAN_MAP = {
    'A': 1, 'I': 1, 'J': 1, 'Q': 1, 'Y': 1,
    'B': 2, 'K': 2, 'R': 2,
    'C': 3, 'G': 3, 'L': 3, 'S': 3,
    'D': 4, 'M': 4, 'T': 4,
    'E': 5, 'H': 5, 'N': 5, 'X': 5,
    'U': 6, 'V': 6, 'W': 6,
    'O': 7, 'Z': 7,
    'F': 8, 'P': 8
}

MASTER_NUMBERS = {11, 22, 33}
KARMIC_DEBT_NUMBERS = {13, 14, 16, 19}

PLANETARY_RULERS = {
    1: "Sun", 2: "Moon", 3: "Jupiter", 4: "Uranus/Rahu", 5: "Mercury",
    6: "Venus", 7: "Neptune/Ketu", 8: "Saturn", 9: "Mars",
    11: "Higher Self/Intuition", 22: "Master Builder", 33: "Master Healer"
}

def clean_name(name: str) -> str:
    """Removes non-alphabetic characters and converts to uppercase."""
    return re.sub(r'[^a-zA-Z\s]', '', name).upper() # Allow spaces for full names

def get_chaldean_value(char: str) -> int:
    """Returns the Chaldean numerology value for a single character."""
    return CHALDEAN_MAP.get(char.upper(), 0)

def calculate_single_digit(number: int, allow_master_numbers: bool = True) -> int:
    """Reduces a number to a single digit, optionally preserving Master Numbers (11, 22, 33)."""
    
    # First check: if the initial number is a Master Number and allowed, return it directly.
    # This handles cases like 11, 22, 33 directly passed in.
    if allow_master_numbers and number in MASTER_NUMBERS:
        return number
    
    # Reduce until single digit or a Master Number (if allowed) is reached
    while number > 9:
        number = sum(int(digit) for digit in str(number))
        # If a Master Number is reached during reduction AND allowed, stop reduction
        if allow_master_numbers and number in MASTER_NUMBERS:
            break # Stop here, don't reduce further
            
    return number

def calculate_expression_number_with_details(full_name: str) -> Tuple[int, Dict]:
    """Calculates the Expression Number with detailed breakdown and planetary association."""
    total = 0
    letter_breakdown = {}
    cleaned_name = clean_name(full_name) # Use the updated clean_name to allow spaces

    # Chaldean map with planetary associations
    CHALDEAN_NUMEROLOGY_MAP_WITH_PLANETS = {
        'A': (1, 'Sun (☉)'), 'J': (1, 'Sun (☉)'), 'Q': (1, 'Sun (☉)'), 'Y': (1, 'Sun (☉)'),
        'B': (2, 'Moon (☽)'), 'K': (2, 'Moon (☽)'), 'R': (2, 'Moon (☽)'),
        'C': (3, 'Jupiter (♃)'), 'G': (3, 'Jupiter (♃)'), 'L': (3, 'Jupiter (♃)'), 'S': (3, 'Jupiter (♃)'),
        'D': (4, 'Rahu (☊)'), 'M': (4, 'Rahu (☊)'), 'T': (4, 'Rahu (☊)'),
        'E': (5, 'Mercury (☿)'), 'H': (5, 'Mercury (☿)'), 'N': (5, 'Mercury (☿)'), 'X': (5, 'Mercury (☿)'),
        'U': (6, 'Venus (♀)'), 'V': (6, 'Venus (♀)'), 'W': (6, 'Venus (♀)'),
        'O': (7, 'Ketu / Neptune (☋)'), 'Z': (7, 'Ketu / Neptune (☋)'),
        'F': (8, 'Saturn (♄)'), 'P': (8, 'Saturn (♄)')
    }
    NUMBER_TO_PLANET_MAP = {
        1: 'Sun (☉)', 2: 'Moon (☽)', 3: 'Jupiter (♃)', 4: 'Rahu (☊)',
        5: 'Mercury (☿)', 6: 'Venus (♀)', 7: 'Ketu / Neptune (☋)', 8: 'Saturn (♄)',
        9: 'Mars (♂)', # 9 is often associated with Mars in some systems
        11: 'Higher Moon / Spiritual Insight', 22: 'Higher Rahu / Master Builder', 33: 'Higher Jupiter / Master Healer'
    }
    
    for letter in cleaned_name:
        if letter in CHALDEAN_NUMEROLOGY_MAP_WITH_PLANETS:
            value, planet = CHALDEAN_NUMEROLOGY_MAP_WITH_PLANETS[letter]
            total += value
            if letter not in letter_breakdown:
                letter_breakdown[letter] = {"value": value, "planet": planet, "count": 0}
            letter_breakdown[letter]["count"] += 1

    reduced = calculate_single_digit(total, True) # Allow master numbers for final expression

    return reduced, {
        "total_before_reduction": total,
        "letter_breakdown": letter_breakdown,
        "is_master_number": reduced in MASTER_NUMBERS,
        "karmic_debt": total in KARMIC_DEBT_NUMBERS,
        "planetary_ruler": NUMBER_TO_PLANET_MAP.get(reduced, 'N/A')
    }

def calculate_life_path_number_with_details(birth_date_str: str) -> Tuple[int, Dict]:
    """
    Calculate Life Path Number (Destiny Number) by adding all digits of complete birth date.
    Master Numbers (11, 22, 33) are NOT reduced.
    """
    
    try:
        year, month, day = map(int, birth_date_str.split('-'))
    except ValueError:
        raise ValueError("Invalid birth date format. Please use YYYY-MM-DD.")

    karmic_debt_present_in_components = []

    # Check for karmic debt in the original numbers before reduction
    if month in KARMIC_DEBT_NUMBERS:
        karmic_debt_present_in_components.append(month)
    if day in KARMIC_DEBT_NUMBERS:
        karmic_debt_present_in_components.append(day)
    
    # Sum all digits from the entire birth date string
    total_sum_all_digits = sum(int(d) for d in re.sub(r'[^0-9]', '', birth_date_str))

    final_number = calculate_single_digit(total_sum_all_digits, True) # Final reduction for Life Path, preserving master numbers

    return final_number, {
        "original_month": month,
        "original_day": day,
        "original_year": year,
        "total_sum_all_digits_before_reduction": total_sum_all_digits,
        "karmic_debt_in_components": list(set(karmic_debt_present_in_components)), # Ensure unique karmic debts
        "is_master_number": final_number in MASTER_NUMBERS
    }
def check_karmic_debt(full_name: str) -> List[int]:
    """
    Checks for karmic debt numbers based on the unreduced sum of the name's Chaldean values.
    Returns a list of karmic debt numbers found.
    """
    cleaned_name = clean_name(full_name)
    total_unreduced = sum(get_chaldean_value(char) for char in cleaned_name)
    
    debts = []
    # Check for karmic debt numbers based on the unreduced sum of the name
    if total_unreduced in KARMIC_DEBT_NUMBERS:
        debts.append(total_unreduced)
    
    # NOTE: This function specifically checks for karmic debt from the *name*.
    # If you also need to check karmic debt from the *birth date*, that logic
    # is typically handled within `calculate_life_path_number_with_details`
    # or `analyze_karmic_lessons` by checking birth date components.
    
    return debts
def calculate_birth_day_number_with_details(birth_date_str: str) -> Tuple[int, Dict]:
    """
    Calculates the Birth Day Number (Primary Ruling Number) from the day of birth.
    Master Numbers (11, 22, 33) are NOT reduced.
    Example: 25th = 2+5=7. 11th = 11.
    """
    if not birth_date_str:
        return 0, {"error": "No birth date provided."}
    try:
        day = int(birth_date_str.split('-')[2])
        if not (1 <= day <= 31):
            raise ValueError("Day must be between 1 and 31.")

        reduced_day = calculate_single_digit(day, True) # Preserve master numbers

        return reduced_day, {
            "original_day": day,
            "is_master_number": reduced_day in MASTER_NUMBERS
        }
    except (ValueError, IndexError) as e:
        logger.error(f"Error calculating Birth Day Number for {birth_date_str}: {e}")
        return 0, {"error": f"Invalid birth day: {e}"}

def calculate_soul_urge_number_with_details(name: str) -> Tuple[int, Dict]:
    """Calculate soul urge (vowels only) using Chaldean map."""
    VOWELS = set('AEIOU')
    total = 0
    vowel_breakdown = {}
    cleaned_name = clean_name(name)

    for letter in cleaned_name:
        if letter in VOWELS:
            value = get_chaldean_value(letter)
            total += value
            vowel_breakdown[letter] = vowel_breakdown.get(letter, 0) + value
    
    reduced = calculate_single_digit(total, True) # Soul Urge can be a Master Number

    return reduced, {
        "total_before_reduction": total,
        "vowel_breakdown": vowel_breakdown,
        "is_master_number": reduced in MASTER_NUMBERS
    }

def calculate_personality_number_with_details(name: str) -> Tuple[int, Dict]:
    """Calculate personality number (consonants only) using Chaldean map."""
    VOWELS = set('AEIOU')
    total = 0
    consonant_breakdown = {}
    cleaned_name = clean_name(name)

    for letter in cleaned_name:
        if letter not in VOWELS and letter != ' ' and letter in get_chaldean_map_keys(): # Consonants and in Chaldean map
            value = get_chaldean_value(letter)
            total += value
            consonant_breakdown[letter] = consonant_breakdown.get(letter, 0) + value

    reduced = calculate_single_digit(total, True) # Personality can be a Master Number

    return reduced, {
        "total_before_reduction": total,
        "consonant_breakdown": consonant_breakdown,
        "is_master_number": reduced in MASTER_NUMBERS
    }

def get_chaldean_map_keys():
    """Helper to get all letters in the Chaldean map."""
    return set('AIJQYBKRCGLSDMTEHNXUVWOFPZ')

def calculate_lo_shu_grid_with_details(birth_date_str: str, suggested_name_expression_num: Optional[int] = None) -> Dict:
    """
    Calculates the Lo Shu Grid digit counts from a birth date.
    Identifies missing numbers and provides basic interpretations.
    Optionally incorporates the suggested name's expression number for analysis.
    """
    try:
        dob_digits = [int(d) for d in birth_date_str.replace('-', '') if d.isdigit()]
    except ValueError:
        raise ValueError("Invalid birth date format for Lo Shu Grid. Please use YYYY-MM-DD.")

    grid_counts = Counter(d for d in dob_digits if 1 <= d <= 9) # Only count digits 1-9 from DOB

    # If a suggested name's expression number is provided, add its single digit to the grid counts
    if suggested_name_expression_num is not None:
        # Reduce master numbers for Lo Shu Grid if they are not already single digits
        # The Lo Shu grid is typically 1-9. Master numbers are usually reduced to their single digit sum for grid analysis.
        grid_friendly_expression = calculate_single_digit(suggested_name_expression_num, allow_master_numbers=False)
        if 1 <= grid_friendly_expression <= 9: # Ensure it's a valid grid digit
            grid_counts[grid_friendly_expression] += 1
            logger.info(f"Lo Shu Grid updated with suggested name's expression number: {grid_friendly_expression}")

    missing_numbers = sorted([i for i in range(1, 10) if i not in grid_counts or grid_counts[i] == 0]) # Ensure 0 count also means missing

    missing_impact = {
        1: "Challenges with independence or self-assertion. Needs to develop leadership.",
        2: "Difficulties with cooperation or sensitivity. Needs to foster diplomacy.",
        3: "Challenges in self-expression or creativity. Needs to communicate more.",
        4: "Potential issues with stability, discipline, or practical matters. Needs structure.",
        5: "Lack of grounding, adaptability, or a need for more freedom. Needs versatility.",
        6: "Harmony issues, challenges with responsibility or nurturing. Needs to serve and care.",
        7: "May indicate a need for deeper introspection or spiritual understanding. Needs wisdom.",
        8: "Potential struggles with material abundance or executive ability. Needs power and organization.",
        9: "Challenges with humanitarianism, compassion, or completion. Needs universal love."
    }

    detailed_missing_lessons = [
        {"number": num, "impact": missing_impact.get(num, "General energy imbalance.")}
        for num in missing_numbers
    ]

    return {
        "grid_counts": dict(grid_counts),
        "missing_numbers": missing_numbers,
        "missing_lessons": detailed_missing_lessons,
        "has_5": grid_counts[5] > 0,
        "has_6": grid_counts[6] > 0,
        "has_3": grid_counts[3] > 0,
        "has_8": grid_counts[8] > 0, # For Expression 8 validation
        "grid_updated_by_name": suggested_name_expression_num is not None
    }

# === STRICT NUMEROLOGY FILTER LOGIC START ===

def filter_strictly_valid_suggestions(suggestions, life_path_number, birth_day_number):
    filtered = []
    for name in suggestions:
        expression, raw_sum = calculate_expression_number(name)
        if not expression:
            continue

        is_lucky = is_lucky_number(expression)
        is_not_karmic = not is_karmic_debt_number(raw_sum)
        is_compatible = (
            expression in EXPRESSION_COMPATIBILITY_MAP.get(life_path_number, []) or
            expression in EXPRESSION_COMPATIBILITY_MAP.get(birth_day_number, [])
        )

        if is_lucky and is_not_karmic and is_compatible:
            filtered.append(name)

    return filtered

def get_conceptual_astrological_data(birth_date: str, birth_time: Optional[str], birth_place: Optional[str]) -> Dict[str, Any]:
    """
    Generates conceptual astrological data.
    In a real application, this would integrate with an astrology API or library.
    """
    logger.warning("Astrological integration for Ascendant is conceptual. Providing simplified data.")
    logger.warning("Astrological integration for Moon Sign is conceptual. Providing simplified data.")
    logger.warning("Astrological integration for Planetary Lords and Degrees is conceptual. Providing simulated data.")

    # Simplified conceptual mapping for Ascendant/Lagna and Moon Sign
    ascendant_sign = "Not Calculated"
    ascendant_ruler = "N/A"
    moon_sign = "Not Calculated"
    moon_ruler = "N/A"
    
    # Conceptual Ascendant (Lagna) - Very simplified mapping based on birth_time (if available)
    if birth_time:
        try:
            birth_hour = int(birth_time.split(':')[0])
            # Simplified conceptual mapping of hour to Ascendant sign and ruler
            ruler_map = {
                0: ("Capricorn", "Saturn (♄)"), 2: ("Aquarius", "Saturn (♄)"),
                4: ("Pisces", "Jupiter (♃)"), 6: ("Aries", "Mars (♂)"),
                8: ("Taurus", "Venus (♀)"), 10: ("Gemini", "Mercury (☿)"),
                12: ("Cancer", "Moon (☽)"), 14: ("Leo", "Sun (☉)"),
                16: ("Virgo", "Mercury (☿)"), 18: ("Libra", "Venus (♀)"),
                20: ("Scorpio", "Mars (♂)"), 22: ("Sagittarius", "Jupiter (♃)")
            }
            # Find the closest hour mapping
            closest_hour = min(ruler_map.keys(), key=lambda h: abs(h - birth_hour))
            ascendant_sign, ascendant_ruler = ruler_map[closest_hour]
        except (ValueError, IndexError):
            pass # Keep defaults if parsing fails

    # Conceptual Moon Sign - Very simplified mapping based on birth_date
    try:
        day = int(birth_date.split('-')[2])
        moon_ruler_map = {
            1: ("Aries", "Mars (♂)"), 2: ("Taurus", "Venus (♀)"), 3: ("Gemini", "Mercury (☿)"),
            4: ("Cancer", "Moon (☽)"), 5: ("Leo", "Sun (☉)"), 6: ("Virgo", "Mercury (☿)"),
            7: ("Libra", "Venus (♀)"), 8: ("Scorpio", "Mars (♂)"), 9: ("Sagittarius", "Jupiter (♃)"),
            10: ("Capricorn", "Saturn (♄)"), 11: ("Aquarius", "Saturn (♄)"), 12: ("Pisces", "Jupiter (♃)")
        }
        # Use month for moon sign for a different conceptual mapping
        month_for_moon = int(birth_date.split('-')[1])
        moon_sign, moon_ruler = moon_ruler_map.get(month_for_moon % 12 + 1, ("Unknown", "N/A")) # Cycle through 1-12
    except (ValueError, IndexError):
        pass # Keep defaults if parsing fails

    # Simulate planetary lords (benefic/malefic) and their conceptual degrees.
    # This is a highly simplified and arbitrary simulation.
    planetary_lords_list = []
    planets = ['Sun (☉)', 'Moon (☽)', 'Mars (♂)', 'Mercury (☿)', 'Jupiter (♃)', 'Venus (♀)', 'Saturn (♄)', 'Rahu (☊)', 'Ketu (☋)']
    for planet in planets:
        nature = random.choice(['Benefic', 'Malefic', 'Neutral'])
        degree = f"{random.randint(0, 29)}° {random.randint(0, 59)}'"
        planetary_lords_list.append({"planet": planet, "nature": nature, "degree": degree})

    return {
        "ascendant_info": {
            "sign": ascendant_sign,
            "ruler": ascendant_ruler,
            "notes": "Conceptual calculation based on birth hour. For precise astrological readings, consult a professional astrologer."
        },
        "moon_sign_info": {
            "sign": moon_sign,
            "ruler": moon_ruler,
            "notes": "Conceptual calculation based on birth date. For precise astrological readings, consult a professional astrologer."
        },
        "planetary_lords": planetary_lords_list,
        "planetary_compatibility": { # This will be filled by check_planetary_compatibility
            "expression_planet": "N/A",
            "compatibility_flags": []
        }
    }

def check_planetary_compatibility(expression_number: int, astro_info: Dict) -> Dict:
    """
    Checks compatibility between expression number's planetary ruler and conceptual astrological info.
    Based on "Influence on Name Correction" and "Match Name Number → Planet" from PDF.
    """
    NUMBER_TO_PLANET_MAP = {
        1: 'Sun (☉)', 2: 'Moon (☽)', 3: 'Jupiter (♃)', 4: 'Rahu (☊)',
        5: 'Mercury (☿)', 6: 'Venus (♀)', 7: 'Ketu / Neptune (☋)', 8: 'Saturn (♄)',
        9: 'Mars (♂)', # 9 is often associated with Mars in some systems
        11: 'Higher Moon / Spiritual Insight', 22: 'Higher Rahu / Master Builder', 33: 'Higher Jupiter / Master Healer'
    }
    expression_planet = NUMBER_TO_PLANET_MAP.get(expression_number)

    compatibility_flags = []

    ascendant_info = astro_info.get('ascendant_info', {})
    planetary_lords = astro_info.get('planetary_lords', [])

    ascendant_ruler = ascendant_info.get('ruler')

    # Rule 1: Harmony with Ascendant's Ruling Planet (from PDF)
    favorable_map_by_ascendant_ruler = {
        'Sun (☉)': [1], 'Moon (☽)': [2], 'Jupiter (♃)': [3], 'Rahu (☊)': [4],
        'Mercury (☿)': [5], 'Venus (♀)': [6], 'Ketu / Neptune (☋)': [7],
        'Saturn (♄)': [8], 'Mars (♂)': [9]
    }

    if ascendant_ruler and ascendant_ruler != "Not Calculated" and ascendant_ruler != "N/A":
        if expression_number in favorable_map_by_ascendant_ruler.get(ascendant_ruler, []):
            if (expression_number == 8 and ascendant_ruler == 'Saturn (♄)') or \
               (expression_number == 4 and ascendant_ruler == 'Rahu (☊)') or \
               (expression_number == 9 and ascendant_ruler == 'Mars (♂)'):
                compatibility_flags.append(f"Expression {expression_number} ({expression_planet}) aligns with {ascendant_ruler}-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}), but requires careful consideration or strong chart support as per numerological tradition.")
            else:
                compatibility_flags.append(f"Expression {expression_number} ({expression_planet}) harmonizes well with your Ascendant ruler ({ascendant_ruler}), amplifying positive traits.")

        # Check for specific conflicts mentioned in PDF
        if ascendant_ruler == 'Sun (☉)' and expression_number in [8, 4]:
            compatibility_flags.append(f"Conflict: Expression {expression_number} ({expression_planet}) is generally avoided with a Sun-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}) due to energetic friction.")
        if ascendant_ruler == 'Moon (☽)' and expression_number == 8:
            compatibility_flags.append(f"Conflict: Expression {expression_number} (Saturn) is generally avoided with a Moon-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}).")
        if ascendant_ruler == 'Jupiter (♃)' and expression_number in [4, 8]:
            compatibility_flags.append(f"Caution: Expression {expression_number} ({expression_planet}) may conflict with a Jupiter-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}) unless specifically supported by the chart.")
        if ascendant_ruler == 'Mercury (☿)' and expression_number in [4, 8]:
            compatibility_flags.append(f"Caution: Expression {expression_number} ({expression_planet}) may conflict with a Mercury-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}).")
        if ascendant_ruler == 'Venus (♀)' and expression_number == 8:
            compatibility_flags.append(f"Caution: Expression {expression_number} (Saturn) may conflict with a Venus-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}).")
        if ascendant_ruler == 'Ketu / Neptune (☋)' and expression_number == 9:
             compatibility_flags.append(f"Caution: Expression {expression_number} (Mars) may conflict with a Ketu/Neptune-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}).")

    # Rule 2: Planetary Conflicts/Support from conceptual chart (from PDF)
    if expression_number == 8 and any(p['planet'] == 'Saturn (♄)' and p['nature'] == 'Malefic' for p in planetary_lords):
        compatibility_flags.append("Caution: Expression 8 (Saturn) may be challenging if your conceptual chart indicates a debilitated Saturn, potentially amplifying obstacles.")

    if expression_number == 3 and any(p['planet'] == 'Jupiter (♃)' and p['nature'] == 'Benefic' for p in planetary_lords):
        compatibility_flags.append("Strong alignment: Expression 3 (Jupiter) is amplified by a favored Jupiter in your conceptual chart, enhancing creativity and wisdom.")

    if expression_number == 9 and any(p['planet'] == 'Mars (♂)' and p['nature'] == 'Benefic' for p in planetary_lords):
        compatibility_flags.append("Strong alignment: Expression 9 (Mars) is amplified by a favored Mars, enhancing courage and drive for humanitarian action.")
    if expression_number == 9 and any(p['planet'] == 'Mars (♂)' and p['nature'] == 'Malefic' for p in planetary_lords):
        compatibility_flags.append("Caution: Expression 9 (Mars) may bring aggressive or conflict-prone energy if your conceptual chart indicates a malefic Mars.")

    if expression_number == 6 and any(p['planet'] == 'Venus (♀)' and p['nature'] == 'Benefic' for p in planetary_lords):
        compatibility_flags.append("Strong alignment: Expression 6 (Venus) is amplified by a favored Venus, enhancing harmony, love, and artistic expression.")
    if expression_number == 6 and any(p['planet'] == 'Venus (♀)' and p['nature'] == 'Malefic' for p in planetary_lords):
        compatibility_flags.append("Caution: Expression 6 (Venus) may bring challenges in relationships or domestic life if your conceptual chart indicates a malefic Venus.")

    if expression_number == 5 and any(p['planet'] == 'Mercury (☿)' and p['nature'] == 'Benefic' for p in planetary_lords):
        compatibility_flags.append("Strong alignment: Expression 5 (Mercury) is amplified by a favored Mercury, enhancing communication, adaptability, and intellectual pursuits.")
    if expression_number == 5 and any(p['planet'] == 'Mercury (☿)' and p['nature'] == 'Malefic' for p in planetary_lords):
        compatibility_flags.append("Caution: Expression 5 (Mercury) may bring instability or communication issues if your conceptual chart indicates a malefic Mercury.")

    if expression_number in [2,7] and any(p['planet'] == 'Moon (☽)' and p['nature'] == 'Benefic' for p in planetary_lords):
        compatibility_flags.append(f"Strong alignment: Expression {expression_number} (Moon/Ketu) is amplified by a favored Moon, enhancing intuition and emotional balance.")
    if expression_number in [2,7] and any(p['planet'] == 'Moon (☽)' and p['nature'] == 'Malefic' for p in planetary_lords):
        compatibility_flags.append(f"Caution: Expression {expression_number} (Moon/Ketu) may bring emotional volatility if your conceptual chart indicates a malefic Moon.")

    if expression_number == 1 and any(p['planet'] == 'Sun (☉)' and p['nature'] == 'Benefic' for p in planetary_lords):
        compatibility_flags.append("Strong alignment: Expression 1 (Sun) is amplified by a favored Sun, enhancing leadership and vitality.")
    if expression_number == 1 and any(p['planet'] == 'Sun (☉)' and p['nature'] == 'Malefic' for p in planetary_lords):
        compatibility_flags.append("Caution: Expression 1 (Sun) may bring ego challenges, arrogance, or difficulties in asserting yourself if your conceptual chart indicates a malefic Sun.")


    return {
        "expression_planet": expression_planet,
        "compatibility_flags": list(set(compatibility_flags)) # Remove duplicates
    }

# Modified: Removed desired_outcome parameter
def get_phonetic_vibration_analysis(full_name: str) -> Dict[str, Any]:
    """
    Performs a conceptual phonetic vibration analysis.
    In a real application, this would involve advanced phonetics and linguistic analysis.
    """
    logger.warning("Phonetic vibration analysis is conceptual. Returning dummy data with more detail.")
    
    cleaned_name = clean_name(full_name).replace(" ", "") # Remove spaces for phonetic analysis

    vowel_count = sum(1 for char in cleaned_name if char.upper() in "AEIOU")
    consonant_count = len(cleaned_name) - vowel_count
    
    vibration_score = 0.7 # Start with a good score
    vibration_notes = []
    qualitative_description = ""

    if len(cleaned_name) < 4:
        vibration_notes.append("Very short names may have less complex vibrational patterns.")
        qualitative_description += "The name is concise and direct. "
    
    if len(cleaned_name) > 0:
        vowel_ratio = vowel_count / len(cleaned_name)
        if vowel_ratio < 0.3 or vowel_ratio > 0.6: # Too few or too many vowels (conceptual range)
            vibration_notes.append(f"Vowel-to-consonant balance ({vowel_ratio:.2f} vowel ratio) might affect the pleasantness of the flow.")
            vibration_score -= 0.1
            if vowel_ratio < 0.3:
                qualitative_description += "Its sound is more consonant-heavy, lending it a grounded and perhaps assertive quality. "
            else:
                qualitative_description += "Its sound is more vowel-heavy, giving it a flowing and open quality. "

    # Simple detection of "harsh" consonant combinations (conceptual)
    harsh_combinations = ["TH", "SH", "CH", "GH", "PH", "CK", "TCH", "DGE", "GHT", "STR", "SCR", "SPL"] # Expanded examples
    
    # Corrected harsh_flags logic:
    harsh_flags = []
    for combo in harsh_combinations:
        if combo in cleaned_name:
            harsh_flags.append(combo)

    if harsh_flags:
        vibration_notes.append(f"Contains potentially harsh consonant combinations: {', '.join(harsh_flags)}. This might affect the overall sound vibration.")
        vibration_score -= 0.2
        qualitative_description += "Some phonetic combinations might create a strong or slightly abrupt impression. "
    
    # Check for pleasant vowel flow (conceptual)
    if re.search(r'(AA|EE|II|OO|UU){2,}', cleaned_name): # Double vowels
         vibration_notes.append("Repeated consecutive vowels might create a drawn-out or emphasized sound.")
         qualitative_description += "The presence of repeated vowels gives it a sustained and perhaps melodious feel. "
         vibration_score -= 0.05
    
    # Removed: Logic that conditionally changes qualitative_description based on desired_outcome

    # General positive note if no specific issues found and no detailed description yet
    if not vibration_notes and not qualitative_description:
        qualitative_description = "The name appears to have a harmonious and balanced phonetic vibration with a pleasant flow, suitable for general positive interactions."
        vibration_score = 0.9 # Excellent
    elif not qualitative_description: # If notes but no description yet
         qualitative_description = "The name has a distinct phonetic quality. " + (vibration_notes[0] if vibration_notes else "")

    return {
        "score": max(0.1, min(1.0, vibration_score)), # Keep score between 0.1 and 1.0
        "notes": list(set(vibration_notes)), # Remove duplicates
        "is_harmonious": vibration_score > 0.65,
        "qualitative_description": qualitative_description.strip()
    }

# Modified: Removed desired_outcome parameter from function signature
@cached_operation(timeout=3600)
async def get_comprehensive_numerology_profile(full_name: str, birth_date: str, birth_time: Optional[str] = None, birth_place: Optional[str] = None, suggested_name_expression_num: Optional[int] = None) -> Dict:
    """
    Get comprehensive numerology profile with caching and advanced calculations.
    Now accepts suggested_name_expression_num to update Lo Shu Grid for validation context.
    """
    
    # Calculate all core numbers with details
    expression_num, expression_details = calculate_expression_number_with_details(full_name)
    life_path_num, life_path_details = calculate_life_path_number_with_details(birth_date)
    soul_urge_num, soul_urge_details = calculate_soul_urge_number_with_details(full_name)
    personality_num, personality_details = calculate_personality_number_with_details(full_name)
    birth_day_num, birth_day_details = calculate_birth_day_number_with_details(birth_date)

    # Calculate Lo Shu Grid, potentially incorporating suggested name's expression
    lo_shu_grid = calculate_lo_shu_grid_with_details(birth_date, suggested_name_expression_num)
    
    # New: Conceptual Astrological Integration
    astro_info = get_conceptual_astrological_data(birth_date, birth_time, birth_place)
    
    # Pass astro_info for planetary compatibility check
    planetary_compatibility = check_planetary_compatibility(expression_num, astro_info)
    astro_info['planetary_compatibility'] = planetary_compatibility # Update astro_info with compatibility flags

    # New: Conceptual Phonetic Vibration - Removed desired_outcome from call
    phonetic_vibration = get_phonetic_vibration_analysis(full_name)

    # New: Edge Case Handling - Pass entire profile data for comprehensive analysis
    # Removed desired_outcome from profile_for_edge_cases
    profile_for_edge_cases = {
        "expression_number": expression_num,
        "life_path_number": life_path_num,
        "birth_day_number": birth_day_num,
        "lo_shu_grid": lo_shu_grid,
        # "desired_outcome": desired_outcome, # Removed
        "ascendant_info": astro_info.get('ascendant_info', {}),
        "moon_sign_info": astro_info.get('moon_sign_info', {}),
        "planetary_lords": astro_info.get('planetary_lords', []),
        "planetary_compatibility": planetary_compatibility,
        "life_path_details": life_path_details, # Needed for karmic debt from birth date
        "expression_details": expression_details # Needed for karmic debt from name
    }
    edge_cases = analyze_edge_cases(profile_for_edge_cases)

    # Calculate compatibility and insights (existing)
    compatibility_insights = calculate_number_compatibility(expression_num, life_path_num)
    karmic_lessons = analyze_karmic_lessons(full_name, birth_date) # Updated to take birth_date for Lo Shu karmic lessons

    # Generate timing recommendations and yearly forecast
    timing_recommendations = generate_timing_recommendations({
        "birth_date": birth_date,
        "expression_number": expression_num,
        "life_path_number": life_path_num
    })
    yearly_forecast = generate_yearly_forecast({
        "birth_date": birth_date,
        "expression_number": expression_num,
        "life_path_number": life_path_num
    })
    
    # Identify potential challenges and development recommendations
    potential_challenges_data = identify_potential_challenges({
        "expression_number": expression_num,
        "life_path_number": life_path_num
    })
    development_recommendations = get_development_recommendations({
        "expression_number": expression_num,
        "life_path_number": life_path_num,
        "karmic_lessons": karmic_lessons # Pass karmic lessons data
    })
    uniqueness_score_data = calculate_uniqueness_score({
        "expression_number": expression_num,
        "life_path_number": life_path_num,
        "birth_day_number": birth_day_num
    })
    
    # Predict success areas
    success_areas_data = predict_success_areas({
        "expression_number": expression_num,
        "life_path_number": life_path_num
    })


    return {
        "full_name": full_name,
        "birth_date": birth_date,
        "birth_time": birth_time,
        "birth_place": birth_place,
        # "desired_outcome": desired_outcome, # Removed from profile data
        "expression_number": expression_num,
        "expression_details": expression_details,
        "life_path_number": life_path_num,
        "life_path_details": life_path_details,
        "birth_day_number": birth_day_num,
        "birth_day_details": birth_day_details,
        "soul_urge_number": soul_urge_num,
        "soul_urge_details": soul_urge_details,
        "personality_number": personality_num,
        "personality_details": personality_details,
        "lo_shu_grid": lo_shu_grid,
        "astro_info": astro_info, # Contains all astro details including compatibility
        "phonetic_vibration": phonetic_vibration,
        "edge_cases": edge_cases,
        "compatibility_insights": compatibility_insights,
        "karmic_lessons": karmic_lessons,
        "timing_recommendations": timing_recommendations, # Added
        "yearly_forecast": yearly_forecast, # Added
        "potential_challenges": potential_challenges_data, # Added
        "development_recommendations": development_recommendations, # Added
        "uniqueness_score": uniqueness_score_data, # Added
        "success_areas": success_areas_data, # Added
        # Removed desired_outcome from hash calculation as well
        "profile_hash": hashlib.md5(f"{full_name}{birth_date}{birth_time}{birth_place}".encode()).hexdigest()
    }

def calculate_number_compatibility(expression: int, life_path: int) -> Dict:
    """Calculate compatibility between expression and life path numbers"""
    # Expanded compatibility scores for more nuance
    compatibility_scores = {
        (1, 1): 0.9, (1, 2): 0.6, (1, 3): 0.9, (1, 4): 0.5, (1, 5): 0.8,
        (1, 6): 0.7, (1, 7): 0.4, (1, 8): 0.9, (1, 9): 0.7, (1, 11): 0.85, (1, 22): 0.8, (1, 33): 0.75,

        (2, 2): 0.9, (2, 3): 0.7, (2, 4): 0.8, (2, 5): 0.5, (2, 6): 0.9,
        (2, 7): 0.8, (2, 8): 0.6, (2, 9): 0.8, (2, 11): 0.95, (2, 22): 0.85, (2, 33): 0.9,

        (3, 3): 0.9, (3, 4): 0.6, (3, 5): 0.9, (3, 6): 0.8, (3, 7): 0.7,
        (3, 8): 0.7, (3, 9): 0.9, (3, 11): 0.75, (3, 22): 0.65, (3, 33): 0.85,

        (4, 4): 0.9, (4, 5): 0.6, (4, 6): 0.8, (4, 7): 0.9, (4, 8): 0.7,
        (4, 9): 0.6, (4, 11): 0.7, (4, 22): 0.95, (4, 33): 0.75,

        (5, 5): 0.9, (5, 6): 0.6, (5, 7): 0.8, (5, 8): 0.7, (5, 9): 0.7,
        (5, 11): 0.8, (5, 22): 0.7, (5, 33): 0.6,

        (6, 6): 0.9, (6, 7): 0.7, (6, 8): 0.8, (6, 9): 0.9, (6, 11): 0.9,
        (6, 22): 0.8, (6, 33): 0.95,

        (7, 7): 0.9, (7, 8): 0.6, (7, 9): 0.8, (7, 11): 0.95, (7, 22): 0.85,
        (7, 33): 0.9,

        (8, 8): 0.9, (8, 9): 0.7, (8, 11): 0.7, (8, 22): 0.95, (8, 33): 0.8,

        (9, 9): 0.9, (9, 11): 0.8, (9, 22): 0.7, (9, 33): 0.95,

        (11, 11): 0.95, (11, 22): 0.85, (11, 33): 0.9,
        (22, 22): 0.95, (22, 33): 0.9,
        (33, 33): 0.95
    }

    key = tuple(sorted((expression, life_path)))
    score = compatibility_scores.get(key, 0.6) # Default to moderate if not explicitly defined

    synergy_areas = []
    if score > 0.85:
        synergy_areas = ["Exceptional harmony", "Powerful synergy", "Effortless understanding", "Accelerated growth"]
    elif score > 0.7:
        synergy_areas = ["Strong compatibility", "Complementary strengths", "Mutual support", "Positive dynamics"]
    elif score > 0.5:
        synergy_areas = ["Moderate compatibility", "Growth opportunities", "Areas for conscious effort", "Dynamic balance"]
    else:
        synergy_areas = ["Learning experiences", "Character building", "Requires conscious effort", "Potential friction points"]

    return {
        "compatibility_score": score,
        "synergy_areas": synergy_areas,
        "description": f"Your Expression Number {expression} and Life Path {life_path} create a {'highly harmonious' if score > 0.85 else 'strong' if score > 0.7 else 'moderate' if score > 0.5 else 'challenging'} compatibility."
    }

def analyze_karmic_lessons(name: str, birth_date_str: str) -> Dict:
    """
    Analyze karmic lessons from missing numbers in name (Chaldean) and Lo Shu Grid.
    This now combines both aspects for a holistic view.
    """
    
    # Karmic lessons from name (missing numbers in Chaldean letter values)
    name_numbers_present = set()
    for letter in clean_name(name):
        if letter in CHALDEAN_MAP: # Use the global CHALDEAN_MAP
            name_numbers_present.add(CHALDEAN_MAP[letter])
    
    missing_from_name = sorted(list(set(range(1, 9)) - name_numbers_present)) # Chaldean only uses 1-8 for letters
    
    # Karmic lessons from Lo Shu Grid (missing numbers in birth date digits)
    lo_shu_grid_data = calculate_lo_shu_grid_with_details(birth_date_str)
    missing_from_lo_shu = lo_shu_grid_data.get('missing_numbers', [])

    # Identify karmic debt numbers from the birth date components (unreduced)
    birth_date_karmic_debts = []
    try:
        year, month, day = map(int, birth_date_str.split('-'))
        if month in KARMIC_DEBT_NUMBERS: birth_date_karmic_debts.append(month)
        if day in KARMIC_DEBT_NUMBERS: birth_date_karmic_debts.append(day)
        # For year, check sum of digits if it's a KD number
        year_sum_digits = sum(int(d) for d in str(year))
        if year_sum_digits in KARMIC_DEBT_NUMBERS: birth_date_karmic_debts.append(year_sum_digits)
    except Exception as e:
        logger.warning(f"Could not check birth date components for karmic debts: {e}")


    karmic_lessons_map = {
        1: "Developing independence and leadership, cultivating self-reliance and initiative.",
        2: "Fostering cooperation, patience, and diplomacy, enhancing sensitivity and balance.",
        3: "Cultivating creativity and communication, embracing self-expression and joy.",
        4: "Building discipline, organization, and stability, focusing on practical matters and hard work.",
        5: "Embracing change, freedom, and adaptability, fostering versatility and exploration.",
        6: "Accepting responsibility and nurturing, creating harmony and selfless service.",
        7: "Seeking spiritual understanding, engaging in introspection and acquiring wisdom.",
        8: "Mastering material world success, developing abundance, power, and executive ability.",
        9: "Developing universal compassion, embracing humanitarianism and completion."
    }
    
    all_missing_numbers = sorted(list(set(missing_from_name + missing_from_lo_shu)))
    
    lessons_list = [
        {"number": num, "lesson": karmic_lessons_map.get(num, "General energy imbalance.")}
        for num in all_missing_numbers
    ]
    
    return {
        "missing_numbers_combined": all_missing_numbers,
        "lessons_summary": lessons_list,
        "birth_date_karmic_debts": list(set(birth_date_karmic_debts)), # Unique karmic debts from birth date
        "priority_lesson": lessons_list[0]['lesson'] if lessons_list else "All core lessons integrated or no significant missing numbers."
    }

def generate_timing_recommendations(profile: Dict) -> Dict:
    """Generate optimal timing recommendations"""
    current_year = datetime.datetime.now().year
    
    # Use client's birth month and day for Personal Year calculation
    try:
        birth_month = int(profile['birth_date'].split('-')[1])
        birth_day = int(profile['birth_date'].split('-')[2])
    except (ValueError, IndexError):
        # Fallback to current month/day if birth date parsing fails (though it should be valid from frontend)
        birth_month = datetime.date.today().month
        birth_day = datetime.date.today().day
        logger.warning("Could not parse birth month/day from profile for Personal Year. Using current month/day as fallback.")

    personal_year_calc_month = calculate_single_digit(birth_month, True)
    personal_year_calc_day = calculate_single_digit(birth_day, True)
    personal_year_calc_year = calculate_single_digit(current_year, True)
    
    personal_year_sum = personal_year_calc_month + personal_year_calc_day + personal_year_calc_year
    personal_year = calculate_single_digit(personal_year_sum, True) # Final reduction for Personal Year

    optimal_activities = {
        1: ["Initiating new projects", "Taking leadership roles", "Embracing independence", "Starting fresh"],
        2: ["Forming partnerships", "Collaborating on projects", "Nurturing relationships", "Seeking balance"],
        3: ["Creative expression", "Socializing", "Communicating ideas", "Enjoying life"],
        4: ["Building foundations", "Organizing finances", "Working diligently", "Establishing security"],
        5: ["Embracing change", "Traveling", "Seeking new experiences", "Adapting to new situations"],
        6: ["Focusing on family/home", "Providing service", "Nurturing self and others", "Cultivating harmony"],
        7: ["Introspection", "Study and research", "Seeking inner truth"],
        8: ["Material achievement", "Business expansion", "Financial management", "Exercising power responsibly"],
        9: ["Completion of cycles", "Humanitarian efforts", "Letting go of the past", "Universal compassion"],
        11: ["Spiritual insight", "Inspiring others", "Intuitive breakthroughs", "Mastering higher vibrations"],
        22: ["Large-scale building", "Manifesting big dreams", "Global impact", "Practical idealism"],
        33: ["Compassionate service", "Healing", "Teaching universal love", "Selfless dedication"]
    }
    
    # Best months map (conceptual, often aligns with the number itself or its reduced form)
    best_months_map = {
        1: [1, 10], 2: [2, 11], 3: [3, 12], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9],
        11: [2, 11], 22: [4], 33: [6]
    }
    
    recommended_months = best_months_map.get(personal_year, [1, 7]) # Default if not found
    
    return {
        "current_personal_year": personal_year,
        "optimal_activities": optimal_activities.get(personal_year, ["General growth and development"]),
        "energy_description": f"Your Personal Year {personal_year} resonates with the energy of {optimal_activities.get(personal_year, ['growth'])[0].lower()}.",
        "best_months_for_action": recommended_months,
    }

# --- Analytics and Insights Helper Functions ---

def get_mitigation_strategies(expression: int) -> List[str]:
    """Get strategies to mitigate challenges"""
    strategies = {
        1: ["Practice patience and listening", "Delegate tasks effectively", "Cultivate empathy and understanding"],
        2: ["Build self-confidence and assertiveness", "Practice decisive action", "Establish clear personal boundaries"],
        3: ["Focus and prioritize energy", "Develop discipline in creative pursuits", "Cultivate deeper, more meaningful relationships"],
        4: ["Embrace flexibility and adaptability", "Take regular breaks to avoid burnout", "Express creativity outside of work"],
        5: ["Restlessness", "Commitment issues", "Impulsiveness", "Overindulgence"],
        6: ["Set healthy boundaries and learn to say no", "Prioritize self-care and personal needs", "Avoid over-commitment and martyrdom"],
        7: ["Engage in practical application of knowledge", "Seek community and connection", "Balance introspection with external engagement"],
        8: ["Balance work-life commitments", "Nurture personal relationships", "Practice generosity and philanthropy", "Focus on ethical power usage"],
        9: ["Manage emotional volatility through mindfulness", "Accept imperfections in others and self", "Practice self-compassion and forgiveness", "Set clear boundaries"],
        11: ["Practice grounding exercises (e.g., walking in nature)", "Manage nervous energy through mindfulness and meditation", "Set realistic and achievable goals, breaking down grand visions"],
        22: ["Delegate tasks effectively to avoid overwhelm", "Prioritize self-care and rest", "Break down large goals into smaller, manageable steps", "Seek practical mentorship"],
        33: ["Learn to set boundaries for emotional well-being", "Balance selfless service with self-care", "Avoid martyrdom and taking on others' burdens excessively"]
    }
    return strategies.get(expression, ["Practice mindfulness", "Seek balance", "Stay grounded"])


def get_growth_opportunities(expression: int) -> List[str]:
    """Get growth opportunities based on expression number"""
    opportunities = {
        1: ["Leadership development", "Mentoring others", "Innovation projects", "Pioneering new ventures"],
        2: ["Conflict resolution", "Team building", "Emotional intelligence", "Diplomacy and negotiation"],
        3: ["Public speaking", "Creative projects", "Social networking", "Expressive arts"],
        4: ["Project management", "Systems thinking", "Financial planning", "Building lasting structures"],
        5: ["Cultural exploration", "Technology adoption", "Network expansion", "Dynamic problem-solving"],
        6: ["Community service", "Healing arts", "Family development", "Nurturing environments"],
        7: ["Research projects", "Spiritual studies", "Analytical work", "Deep philosophical inquiry"],
        8: ["Business development", "Financial management", "Strategic planning", "Philanthropy and large-scale impact"],
        9: ["Global initiatives", "Teaching opportunities", "Humanitarian work", "Universal compassion advocacy"],
        11: ["Spiritual leadership", "Intuitive development", "Inspiring others", "Visionary pursuits"],
        22: ["Large-scale project execution", "Building lasting structures", "Transformational leadership", "Manifesting grand visions"],
        33: ["Global humanitarian efforts", "Teaching universal love", "Healing initiatives", "Selfless service leadership"]
    }
    return opportunities.get(expression, ["Personal development", "Skill building", "Service opportunities"])

def calculate_uniqueness_score(profile: Dict) -> Dict:
    """Calculate how unique this numerological profile is"""
    expression = profile.get('expression_number', 5)
    life_path = profile.get('life_path_number', 5)
    birth_day_number = profile.get('birth_day_number', 0)
    
    rarity_multiplier = 1.0
    rarity_factors = []

    if expression in MASTER_NUMBERS:
        rarity_multiplier += 0.5
        rarity_factors.append(f"Master Expression Number ({expression})")
    if life_path in MASTER_NUMBERS:
        rarity_multiplier += 0.5
        rarity_factors.append(f"Master Life Path Number ({life_path})")
    if birth_day_number in MASTER_NUMBERS:
        rarity_multiplier += 0.3
        rarity_factors.append(f"Master Birth Day Number ({birth_day_number})")

    # Consider rare combinations
    combination_rarity_score = 0.0
    if (expression == 8 and life_path == 1) or (expression == 1 and life_path == 8):
        combination_rarity_score += 0.4
        rarity_factors.append("Expression 8 / Life Path 1 Conflict (Rare Challenge)")
    if (expression in {4,8} and life_path in {3,6,9,11,33}): # Examples of challenging/unique combos
        combination_rarity_score += 0.2
        rarity_factors.append("Challenging core number combination")
    if (expression in MASTER_NUMBERS and life_path in MASTER_NUMBERS): # Master number combos
        combination_rarity_score += 0.3
        rarity_factors.append("Multiple Master Numbers in core profile")

    uniqueness = min(0.95, (combination_rarity_score + 0.3) * rarity_multiplier)
    
    if not rarity_factors:
        rarity_factors.append("Standard numerical pattern")

    return {
        "score": uniqueness,
        "interpretation": "Highly unique and powerful" if uniqueness > 0.85 else "Unique and insightful" if uniqueness > 0.7 else "Common pattern with individual strengths",
        "rarity_factors": rarity_factors
    }

def predict_success_areas(profile: Dict) -> Dict:
    """Predict areas of likely success"""
    expression = profile.get('expression_number', 5)
    life_path = profile.get('life_path_number', 5)
    
    success_areas_map = {
        1: ["Leadership", "Entrepreneurship", "Innovation", "Pioneering new ventures"],
        2: ["Collaboration", "Diplomacy", "Counseling", "Partnerships", "Mediation"],
        3: ["Creative arts", "Communication", "Entertainment", "Public speaking", "Marketing"],
        4: ["Management", "Organization", "Finance", "Construction", "System building"],
        5: ["Travel", "Technology", "Sales", "Journalism", "Public relations"],
        6: ["Healthcare", "Education", "Service", "Family business", "Community work"],
        7: ["Research", "Analysis", "Spirituality", "Science", "Philosophy"],
        8: ["Business", "Financial", "Authority", "Real estate", "Law"],
        9: ["Humanitarian work", "Teaching", "Global reach", "Philanthropy", "Social reform"],
        11: ["Spiritual leadership", "Intuitive development", "Innovation", "Inspiring masses"],
        22: ["Large-scale building", "Transformation", "Leadership in big organizations", "Practical idealism"],
        33: ["Healing", "Teaching", "Service leadership", "Universal love advocacy", "Spiritual guidance"]
    }
    
    primary = success_areas_map.get(expression, ["General success"])
    secondary = success_areas_map.get(life_path, [])
    
    combined = list(set(primary + secondary))
    
    return {
        "primary_areas": primary,
        "secondary_areas": secondary,
        "combined_strengths": combined[:6],
        "confidence_level": "High" if len(combined) > 4 else "Moderate"
    }

def identify_potential_challenges(profile: Dict) -> Dict:
    """Identify potential challenges and growth areas"""
    expression = profile.get('expression_number', 5)
    life_path = profile.get('life_path_number', 5)
    
    challenges_map = {
        1: ["Impatience", "Domination tendency", "Isolation", "Ego challenges"],
        2: ["Over-sensitivity", "Indecision", "Dependency", "Lack of assertiveness"],
        3: ["Scattered energy", "Superficiality", "Inconsistency", "Emotional avoidance"],
        4: ["Rigidity", "Resistance to change", "Workaholic tendencies", "Lack of spontaneity"],
        5: ["Restlessness", "Commitment issues", "Impulsiveness", "Overindulgence"],
        6: ["Over-responsibility", "Interference", "Martyrdom", "Self-sacrifice"],
        7: ["Isolation", "Over-analysis", "Emotional detachment", "Cynicism"],
        8: ["Materialism", "Power struggles", "Relationship neglect", "Authoritarianism"],
        9: ["Emotional volatility", "Disappointment", "Burnout", "Difficulty with boundaries"],
        11: ["Nervous tension", "Unrealistic expectations", "Hypersensitivity", "Self-doubt"],
        22: ["Overwhelming pressure", "Perfectionism", "Burnout risk", "Difficulty delegating"],
        33: ["Emotional overwhelm, taking on others' pain, and tendency toward self-sacrifice", "Martyrdom", "Boundary issues"]
    }
    
    potential_challenges = challenges_map.get(expression, ["General life challenges"])
    # Add challenges from life path if they are distinct
    life_path_challenges = challenges_map.get(life_path, [])
    for challenge in life_path_challenges:
        if challenge not in potential_challenges:
            potential_challenges.append(challenge)

    return {
        "potential_challenges": potential_challenges,
        "mitigation_strategies": get_mitigation_strategies(expression),
        "growth_opportunities": get_growth_opportunities(expression)
    }

def get_development_recommendations(profile: Dict) -> Dict:
    """Get personalized development recommendations"""
    expression = profile.get('expression_number', 5)
    life_path = profile.get('life_path_number', 5)
    karmic_lessons_data = profile.get('karmic_lessons', {})
    karmic_lessons_summary = karmic_lessons_data.get('lessons_summary', [])
    
    # Extract specific lessons for actionable guidance
    karmic_work_items = [lesson['lesson'] for lesson in karmic_lessons_summary]
    
    return {
        "immediate_focus": f"Develop your Expression {expression} energy through {get_growth_opportunities(expression)[0].lower()}",
        "long_term_goal": f"Align with your Life Path {life_path} by embracing {get_growth_opportunities(life_path)[0].lower()}",
        "karmic_work": karmic_work_items[:3] if karmic_work_items else ["Continue spiritual growth and self-awareness."],
        "monthly_practices": [
            "Daily meditation or reflection to connect with inner wisdom.",
            "Journaling about number patterns and their manifestation in your life.",
            f"Actively practicing the positive traits of your Expression {expression} and Life Path {life_path} numbers.",
            "Consciously working on identified challenges and integrating mitigation strategies."
        ]
    }

def generate_yearly_forecast(profile: Dict, years: int = 3) -> Dict:
    """Generate multi-year numerological forecast"""
    current_year = datetime.datetime.now().year
    
    forecast = {}
    
    for year_offset in range(years):
        target_year = current_year + year_offset
        # Personal Year calculation based on birth month and day, and target year
        try:
            birth_month = int(profile['birth_date'].split('-')[1])
            birth_day = int(profile['birth_date'].split('-')[2])
        except (ValueError, IndexError):
            # Fallback to current month/day if birth date parsing fails (though it should be valid from frontend)
            birth_month = datetime.date.today().month
            birth_day = datetime.date.today().day
            logger.warning("Could not parse birth month/day for Personal Year. Using current month/day as fallback.")

        personal_year_calc_month = calculate_single_digit(birth_month, True)
        personal_year_calc_day = calculate_single_digit(birth_day, True)
        personal_year_calc_year = calculate_single_digit(target_year, True)
        
        personal_year_sum = personal_year_calc_month + personal_year_calc_day + personal_year_calc_year
        personal_year = calculate_single_digit(personal_year_sum, True)
        
        year_themes = {
            1: {"theme": "New Beginnings & Initiative", "focus": "Starting fresh, leadership opportunities, self-discovery."},
            2: {"theme": "Cooperation & Partnership", "focus": "Forming alliances, nurturing relationships, seeking balance."},
            3: {"theme": "Creative Expression & Communication", "focus": "Socializing, artistic pursuits, joyful self-expression."},
            4: {"theme": "Building Foundations & Discipline", "focus": "Hard work, organization, establishing security, practical planning."},
            5: {"theme": "Freedom & Dynamic Change", "focus": "Embracing new experiences, travel, adaptability, adventure."},
            6: {"theme": "Responsibility & Harmony", "focus": "Focusing on family/home, providing service, nurturing self and others, community engagement."},
            7: {"theme": "Introspection & Spiritual Growth", "focus": "Study, reflection, inner development, seeking deeper truths."},
            8: {"theme": "Material Success & Power", "focus": "Business expansion, financial management, achievement, leadership in material world."},
            9: {"theme": "Completion & Humanitarianism", "focus": "Endings, letting go of the past, humanitarian efforts, universal compassion."},
            11: {"theme": "Spiritual Illumination & Inspiration", "focus": "Intuitive breakthroughs, inspiring others, spiritual leadership."},
            22: {"theme": "Master Builder Year & Manifestation", "focus": "Large-scale manifestation, practical idealism, global impact."},
            33: {"theme": "Universal Service & Healing", "focus": "Compassionate leadership, healing the world, selfless dedication."}
        }
        
        year_info = year_themes.get(personal_year, {"theme": "General Growth", "focus": "Personal development and learning."})
        
        # Optimal months can be conceptually tied to the personal year number
        optimal_months_for_year = []
        if personal_year == 1: optimal_months_for_year = [1, 10]
        elif personal_year == 2: optimal_months_for_year = [2, 11]
        elif personal_year == 3: optimal_months_for_year = [3, 12]
        elif personal_year == 4: optimal_months_for_year = [4]
        elif personal_year == 5: optimal_months_for_year = [5]
        elif personal_year == 6: optimal_months_for_year = [6]
        elif personal_year == 7: optimal_months_for_year = [7]
        elif personal_year == 8: optimal_months_for_year = [8]
        elif personal_year == 9: optimal_months_for_year = [9]
        elif personal_year == 11: optimal_months_for_year = [2, 11] # Reduced to 2
        elif personal_year == 22: optimal_months_for_year = [4] # Reduced to 4
        elif personal_year == 33: optimal_months_for_year = [6] # Reduced to 6

        forecast[target_year] = {
            "personal_year": personal_year,
            "theme": year_info["theme"],
            "focus_areas": year_info["focus"],
            "optimal_months": optimal_months_for_year,
            "energy_level": "High" if personal_year in [1, 3, 5, 8, 11, 22, 33] else "Moderate" if personal_year in [2, 6, 9] else "Reflective"
        }
    
    return forecast

def analyze_edge_cases(profile_data: Dict) -> List[Dict]:
    """
    Identifies specific challenging numerological combinations and provides insights,
    now incorporating conceptual astrological data and Lo Shu Grid.
    """
    edge_cases = []

    expression_number = profile_data.get('expression_number')
    life_path_number = profile_data.get('life_path_number')
    birth_day_number = profile_data.get('birth_day_number')
    lo_shu_grid = profile_data.get('lo_shu_grid', {})
    # desired_outcome = profile_data.get('desired_outcome', '').lower() # Removed
    astro_info = profile_data.get('astro_info', {})
    planetary_lords = astro_info.get('planetary_lords', [])
    ascendant_info = astro_info.get('ascendant_info', {})
    
    # Common numerical constants
    
    NUMBER_TO_PLANET_MAP = {
        1: 'Sun (☉)', 2: 'Moon (☽)', 3: 'Jupiter (♃)', 4: 'Rahu (☊)',
        5: 'Mercury (☿)', 6: 'Venus (♀)', 7: 'Ketu / Neptune (☋)', 8: 'Saturn (♄)',
        9: 'Mars (♂)', # 9 is often associated with Mars in some systems
        11: 'Higher Moon / Spiritual Insight', 22: 'Higher Rahu / Master Builder', 33: 'Higher Jupiter / Master Healer'
    }

    # Expression = 8, Life Path = 1 (Conflict: Saturn vs Sun) (from PDF)
    if expression_number == 8 and life_path_number == 1:
        edge_cases.append({
            "type": "Expression 8 / Life Path 1 Conflict (Saturn vs. Sun)",
            "description": "A clash between Saturn's discipline (8) and Sun's leadership (1) can lead to internal friction or external power struggles. The 8 may bring delays or heavy lessons to the pioneering 1, especially in areas of authority and recognition.",
            "resolution_guidance": "A name change to Expression 1, 3, 5, or 6 is often recommended to harmonize with the Life Path 1 energy. Focus on balanced leadership, integrity, and avoiding materialism. Cultivate patience and humility."
        })
    
    # Master Number Expression but LP = 4 (Master vs Builder) (from PDF)
    if expression_number in MASTER_NUMBERS and life_path_number == 4:
        edge_cases.append({
            "type": f"Master Expression {expression_number} / Life Path 4 Challenge (Vision vs. Structure)",
            "description": f"The expansive, visionary energy of Master Number {expression_number} can feel constrained or frustrated by the practical, structured nature of Life Path 4. It's a call to ground big visions into tangible reality, which can be a heavy task, and may lead to burnout if not managed.",
            "resolution_guidance": "Focus on disciplined manifestation. If the Master energy feels overwhelming, it may reduce to a single digit (e.g., 11 to 2, 22 to 4, 33 to 6) which might be more harmonious with the LP 4, offering a more practical path. Break down large goals into small, actionable steps."
        })

    # 4 in Expression and Birth Day Number = 4 (Too much karma/rigidity) (from PDF)
    if expression_number == 4 and birth_day_number == 4:
         edge_cases.append({
            "type": "Double 4 (Expression & Birth Day Number - Amplified Karma)",
            "description": "Having both Expression and Birth Day Numbers as 4 can amplify the karmic lessons associated with hard work, discipline, and potential rigidity. It suggests a life path heavily focused on building and structure, which can sometimes feel burdensome and limit flexibility. Astrological validation is highly recommended to understand specific challenges and periods of intense effort."
        })

    # Lo Shu Grid lacks 5 or 6 (Avoid assigning name #5 or #6 to Expression if missing) (from PDF)
    if not lo_shu_grid.get('has_5') and expression_number == 5:
        edge_cases.append({
            "type": "Expression 5 with Missing 5 in Lo Shu Grid (Adaptability Challenge)",
            "description": "A missing 5 in your birth date (Lo Shu Grid) indicates a need to develop adaptability and freedom. Assigning Expression 5 might make it challenging to fully integrate this energy, potentially leading to instability or restlessness.",
            "resolution_guidance": "Focus on developing inner flexibility and grounding practices. While the name aims for adaptability, the innate lack of 5 may require conscious effort to embrace change positively."
        })
    if not lo_shu_grid.get('has_6') and expression_number == 6:
        edge_cases.append({
            "type": "Expression 6 with Missing 6 in Lo Shu Grid (Harmony/Responsibility Challenge)",
            "description": "A missing 6 in your birth date (Lo Shu Grid) can indicate challenges with harmony, responsibility, or nurturing. Assigning Expression 6 might make it difficult to fully embody its compassionate and service-oriented traits.",
            "resolution_guidance": "Focus on developing compassion and responsibility through conscious effort. The name can support this, but foundational work on these themes may be necessary."
        })
    
    # NEW: Lo Shu Grid Balancing for Suggested Name's Expression Number
    # This is not an "edge case" in the negative sense, but a positive alignment
    # that should be highlighted if the suggested name's expression number
    # helps fill a missing number in the Lo Shu Grid.
    if expression_number in lo_shu_grid.get('missing_numbers', []):
        edge_cases.append({
            "type": f"Expression {expression_number} Harmonizes with Missing Lo Shu Grid Number",
            "description": f"The suggested name's Expression Number ({expression_number}) directly addresses and helps balance the missing energy of {expression_number} in your Lo Shu Grid. This promotes the integration of its qualities (e.g., {'adaptability' if expression_number == 5 else 'harmony/responsibility' if expression_number == 6 else 'creativity/communication' if expression_number == 3 else 'leadership/independence' if expression_number == 1 else 'N/A'}) into your life.",
            "resolution_guidance": "Embrace the qualities of this number as they are now reinforced by your name, helping to fill a foundational energetic gap in your blueprint."
        })


    # Removed: Profession-based edge cases as desired_outcome is removed.
    # spiritual_professions = ["healer", "teacher", "counselor", "artist", "researcher", "spiritual guide"]
    # if any(p in desired_outcome for p in spiritual_professions):
    #     if expression_number in {2, 7}:
    #         if not (lo_shu_grid.get('has_5') and lo_shu_grid.get('has_6')):
    #             edge_cases.append({
    #                 "type": f"Spiritual Profession / Expression {expression_number} with Unsupportive Lo Shu Grid",
    #                 "description": f"While Expression {expression_number} aligns with your desired spiritual profession, the missing 5 (grounding) or 6 (harmony/service) in your Lo Shu Grid might make it challenging to fully ground and manifest these energies practically.",
    #                 "resolution_guidance": "Focus on practical application of your spiritual gifts and building stable foundations. Consider names that reinforce grounding numbers if appropriate, or consciously develop the missing energies."
    #             })

    # Karmic Debt Numbers in Birth Date or Original Name
    original_karmic_debts = profile_data.get('life_path_details', {}).get('karmic_debt_in_components', [])
    
    # To check for karmic debt in original name's expression, we need its unreduced total.
    # Assuming `expression_details` contains `total_before_reduction` for the *original* name.
    original_name_expression_total = profile_data.get('expression_details', {}).get('total_before_reduction')
    if original_name_expression_total in KARMIC_DEBT_NUMBERS and original_name_expression_total not in original_karmic_debts:
        original_karmic_debts.append(original_name_expression_total)
    
    if original_karmic_debts:
        unique_debts = list(set(original_karmic_debts))
        edge_cases.append({
            "type": f"Karmic Debt Numbers Present: {', '.join(map(str, unique_debts))}",
            "description": f"These numbers indicate specific life lessons or challenges to overcome. The energy of these debts can manifest as recurring patterns or obstacles until the lessons are learned.",
            "resolution_guidance": "Consciously work on the lessons associated with each karmic debt number (e.g., 13 for hard work, 14 for adaptability, 16 for humility, 19 for independence). A name change can help mitigate but not eliminate these core lessons."
        })

    # Astrological Edge Cases (Conceptual, from PDF)
    expression_planet = NUMBER_TO_PLANET_MAP.get(expression_number)
    ascendant_ruler = ascendant_info.get('ruler')

    # Example: Expression 8 (Saturn) vs. Sun-ruled Ascendant (Leo) - from PDF
    if expression_number == 8 and ascendant_ruler == 'Sun (☉)':
        edge_cases.append({
            "type": "Expression 8 vs. Sun-ruled Ascendant Conflict",
            "description": f"Your Expression Number 8 (ruled by Saturn) may create friction with your Sun-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}). This can manifest as struggles between ambition and personal freedom, or delays in leadership roles. Saturn's restrictive nature can challenge the Sun's expansive drive.",
            "resolution_guidance": "Focus on disciplined effort without rigidity. Cultivate patience and ensure your actions are aligned with your true self, not just material gain. Consider names with Expression 1, 3, 5, or 6 if seeking more direct alignment with solar energy."
        })
    
    # Example: Expression 4 (Rahu) vs. Jupiter-ruled Ascendant (Sagittarius/Pisces) - from PDF
    if expression_number == 4 and ascendant_ruler == 'Jupiter (♃)':
        edge_cases.append({
            "type": "Expression 4 vs. Jupiter-ruled Ascendant Caution",
            "description": f"Expression 4 (Rahu) can bring unconventional paths and sudden changes, which might challenge the wisdom and expansion of a Jupiter-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}). This combination requires careful navigation of material pursuits and spiritual growth.",
            "resolution_guidance": "Embrace structure and practicality (4) but ensure it serves a higher purpose (Jupiter). Avoid impulsive decisions and seek spiritual guidance to navigate challenges. A name with Expression 3 or 9 might offer more harmonious expansion."
        })

    # Example: Debilitated Saturn in conceptual chart and Expression 8 - from PDF
    if expression_number == 8 and any(p['planet'] == 'Saturn (♄)' and p['nature'] == 'Malefic' for p in planetary_lords):
        edge_cases.append({
            "type": "Expression 8 with Challenged Saturn Influence",
            "description": "If your conceptual chart indicates a challenging Saturn influence, an Expression Number of 8 can amplify its difficult aspects, potentially leading to increased delays, obstacles, or feelings of burden related to career and material success. This combination demands immense patience and integrity.",
            "resolution_guidance": "Focus on integrity, patience, and selfless service. Avoid shortcuts and practice detachment from outcomes. Consider names with Expression 1, 3, 5, or 6 to balance this energy, unless your chart strongly supports a powerful 8."
        })

    # Example: Favored Jupiter in conceptual chart and Expression 3 - from PDF
    if expression_number == 3 and any(p['planet'] == 'Jupiter (♃)' and p['nature'] == 'Benefic' for p in planetary_lords):
        edge_cases.append({
            "type": "Expression 3 with Favored Jupiter Influence",
            "description": "An Expression Number of 3 (ruled by Jupiter) combined with a favored Jupiter in your conceptual chart can significantly amplify creativity, optimism, and opportunities for growth, especially in communication, teaching, and expansion. This is a very positive alignment.",
            "resolution_guidance": "Embrace opportunities for self-expression and knowledge sharing. Maintain focus to avoid scattering your energies. Your natural optimism and wisdom will be greatly enhanced. Channel this energy into constructive and uplifting endeavors."
        })
    
    # Add other planetary conflicts/supports as per PDF's "Planetary Conflicts and Support" section
    # If chart favors Mars, a name totaling 9 may amplify positivity.
    if expression_number == 9 and any(p['planet'] == 'Mars (♂)' and p['nature'] == 'Benefic' for p in planetary_lords):
        edge_cases.append({
            "type": "Expression 9 with Favored Mars Influence",
            "description": "Expression 9 (Mars) is amplified by a favored Mars in your conceptual chart, enhancing your drive, courage, and capacity for humanitarian action. This combination supports strong leadership in service.",
            "resolution_guidance": "Channel your energy into purposeful action and avoid aggression. Use your drive for the greater good."
        })
    # If chart has malefic Mars, avoid 9 (implied from PDF's general rule of avoiding conflicts)
    if expression_number == 9 and any(p['planet'] == 'Mars (♂)' and p['nature'] == 'Malefic' for p in planetary_lords):
        edge_cases.append({
            "type": "Expression 9 with Malefic Mars Influence",
            "description": "Expression 9 (Mars) may bring aggressive energy, impulsiveness, or conflict if your conceptual chart indicates a malefic Mars. This can make humanitarian efforts challenging.",
            "resolution_guidance": "Cultivate patience and diplomacy. Focus on constructive outlets for your energy and avoid confrontations. A name with a different Expression number might be more harmonious."
        })

    # If chart favors Venus, a name totaling 6 may amplify positivity.
    if expression_number == 6 and any(p['planet'] == 'Venus (♀)' and p['nature'] == 'Benefic' for p in planetary_lords):
        edge_cases.append({
            "type": "Expression 6 with Favored Venus Influence",
            "description": "Expression 6 (Venus) is amplified by a favored Venus in your conceptual chart, enhancing harmony, creativity, and love in relationships and domestic life. This is highly beneficial for personal and family well-being.",
            "resolution_guidance": "Embrace your nurturing qualities and artistic talents. Create beauty and harmony in your surroundings."
        })
    # If chart has malefic Venus, avoid 6 (implied from PDF's general rule of avoiding conflicts)
    if expression_number == 6 and any(p['planet'] == 'Venus (♀)' and p['nature'] == 'Malefic' for p in planetary_lords):
        edge_cases.append({
            "type": "Expression 6 with Malefic Venus Influence",
            "description": "Expression 6 (Venus) may bring challenges in relationships or domestic life if your conceptual chart indicates a malefic Venus. This can lead to difficulties in nurturing or finding balance.",
            "resolution_guidance": "Focus on self-love and setting healthy boundaries in relationships. Avoid codependency. A name with a different Expression number might be more supportive."
        })

    # If chart favors Mercury, a name totaling 5 may amplify positivity.
    if expression_number == 5 and any(p['planet'] == 'Mercury (☿)' and p['nature'] == 'Benefic' for p in planetary_lords):
        edge_cases.append({
            "type": "Expression 5 with Favored Mercury Influence",
            "description": "Expression 5 (Mercury) is amplified by a favored Mercury in your conceptual chart, enhancing communication, adaptability, and intellectual pursuits. This supports versatility and quick thinking.",
            "resolution_guidance": "Embrace new experiences and opportunities for learning. Use your communication skills effectively. Avoid scattering your energy too widely."
        })
    # If chart has malefic Mercury, avoid 5 (implied from PDF's general rule of avoiding conflicts)
    if expression_number == 5 and any(p['planet'] == 'Mercury (☿)' and p['nature'] == 'Malefic' for p in planetary_lords):
        edge_cases.append({
            "type": "Expression 5 with Malefic Mercury Influence",
            "description": "Expression 5 (Mercury) may bring instability, restlessness, or communication challenges if your conceptual chart indicates a malefic Mercury. This can impact your leadership journey.",
            "resolution_guidance": "Practice grounding and focus. Ensure clear and concise communication. A name with a different Expression number might provide more stability."
        })

    # If chart favors Moon, a name totaling 2 or 7 may amplify positivity.
    if expression_number in [2,7] and any(p['planet'] == 'Moon (☽)' and p['nature'] == 'Benefic' for p in planetary_lords):
        edge_cases.append({
            "type": f"Expression {expression_number} with Favored Moon Influence",
            "description": f"Expression {expression_number} (Moon/Ketu) is amplified by a favored Moon in your conceptual chart, enhancing intuition and emotional balance, and spiritual insight. This supports deep connections and inner peace.",
            "resolution_guidance": "Trust your intuition and nurture your emotional well-being. Embrace your sensitive nature as a strength."
        })
    # If chart has malefic Moon, avoid 2 or 7 (implied from PDF's general rule of avoiding conflicts)
    if expression_number in [2,7] and any(p['planet'] == 'Moon (☽)' and p['nature'] == 'Malefic' for p in planetary_lords):
        edge_cases.append({
            "type": f"Expression {expression_number} with Malefic Moon Influence",
            "description": f"Expression {expression_number} (Moon/Ketu) may bring emotional volatility, mood swings, or difficulties in relationships if your conceptual chart indicates a malefic Moon. This can impact inner peace.",
            "resolution_guidance": "Practice emotional regulation and self-care. Seek stability in your environment. A name with a different Expression number might offer a more emotional resilience."
        })

    # If chart favors Sun, a name totaling 1 may amplify positivity.
    if expression_number == 1 and any(p['planet'] == 'Sun (☉)' and p['nature'] == 'Benefic' for p in planetary_lords):
        edge_cases.append({
            "type": "Expression 1 with Favored Sun Influence",
            "description": "Expression 1 (Sun) is amplified by a favored Sun in your conceptual chart, enhancing leadership qualities, vitality, and self-confidence. This is highly beneficial for pioneering and achieving recognition.",
            "resolution_guidance": "Embrace your leadership potential and shine brightly. Inspire others through your actions."
        })
    # If chart has malefic Sun, avoid 1 (implied from PDF's general rule of avoiding conflicts)
    if expression_number == 1 and any(p['planet'] == 'Sun (☉)' and p['nature'] == 'Malefic' for p in planetary_lords):
        edge_cases.append({
            "type": "Expression 1 with Malefic Sun Influence",
            "description": "Expression 1 (Sun) may bring ego challenges, arrogance, or difficulties in asserting yourself if your conceptual chart indicates a malefic Sun. This can impact your leadership journey.",
            "resolution_guidance": "Practice humility and self-awareness. Focus on service-oriented leadership rather than personal glory. A name with a different Expression number might offer a more balanced approach."
        })

    return edge_cases

# --- Validation Logic for a Single Name ---
# Modified: Removed desired_outcome from function signature and logic
def validate_suggested_name_rules(suggested_name: str, client_profile: Dict) -> Tuple[bool, str]:
    """
    Performs a rule-based validation for a single suggested name.
    Returns (is_valid: bool, rationale: str).
    """
    is_valid = True
    reasons = []

    # Recalculate numbers for the suggested name
    suggested_exp_num, suggested_exp_details = calculate_expression_number_with_details(suggested_name)
    
    # Get client's core numbers
    life_path_num = client_profile.get('life_path_number')
    birth_day_num = client_profile.get('birth_day_number')
    # desired_outcome = client_profile.get('desired_outcome', '').lower() # Removed

    # Get Lo Shu Grid (potentially updated with suggested name's expression for analysis)
    lo_shu_grid_data = calculate_lo_shu_grid_with_details(client_profile.get('birth_date'), suggested_exp_num)
    
    # Get Astro Info
    # Use the full astro_info from the client_profile, which was calculated initially
    astro_info = client_profile.get('astro_info', {}) 
    planetary_compatibility = check_planetary_compatibility(suggested_exp_num, astro_info)

    # Rule 1: Master Numbers (11, 22, 33) - only if supported by Life Path or Birth Day
    if suggested_exp_num in MASTER_NUMBERS:
        # If the suggested name is a Master Number, check if the client's Life Path or Birth Day
        # is also a Master Number or the same number (e.g., LP 2 for Exp 11, LP 4 for Exp 22, LP 6 for Exp 33)
        # This is a common numerological guideline for "grounding" Master Numbers.
        is_master_supported = False
        if life_path_num in MASTER_NUMBERS or \
           (suggested_exp_num == 11 and life_path_num == 2) or \
           (suggested_exp_num == 22 and life_path_num == 4) or \
           (suggested_exp_num == 33 and life_path_num == 6):
            is_master_supported = True
        
        if birth_day_num in MASTER_NUMBERS or \
           (suggested_exp_num == 11 and birth_day_num == 2) or \
           (suggested_exp_num == 22 and birth_day_num == 4) or \
           (suggested_exp_num == 33 and birth_day_num == 6):
            is_master_supported = True # Can be supported by either

        if not is_master_supported:
            is_valid = False
            reasons.append(f"Master Expression {suggested_exp_num} is highly potent but may be challenging to embody without a supporting Master Life Path or Birth Day. Consider if the client is ready for this intense energy.")
        else:
            reasons.append(f"Master Expression {suggested_exp_num} is well-supported by your core numbers, offering immense potential for higher purpose and manifestation.")
    
    # Rule 2: Karmic Debt Numbers (13, 14, 16, 19) - avoid as final Expression (unreduced sum)
    if suggested_exp_details.get('total_before_reduction') in KARMIC_DEBT_NUMBERS:
        is_valid = False
        reasons.append(f"The suggested name's unreduced total ({suggested_exp_details.get('total_before_reduction')}) results in a Karmic Debt Number. This can bring significant life lessons and challenges. It is generally advised to avoid this as a primary Expression Number.")
    
    # Rule 3: Lo Shu Grid Balance - especially for 5, 6, 8
    # If the suggested name's expression number is missing from the Lo Shu Grid, it's a positive.
    if suggested_exp_num in lo_shu_grid_data.get('missing_numbers', []):
        reasons.append(f"The suggested Expression Number {suggested_exp_num} helps to balance the missing energy of {suggested_exp_num} in your Lo Shu Grid, promoting integration of its qualities.")
    
    # If 8 is missing from Lo Shu Grid, Expression 8 is problematic
    if not lo_shu_grid_data.get('has_8') and suggested_exp_num == 8:
        is_valid = False
        reasons.append("Expression 8 (material success, power) is problematic if the number 8 is missing from your Lo Shu Grid, as it indicates a lack of foundational energy for managing this power effectively. This can lead to financial instability or power struggles.")

    # Rule 4: Astrological Compatibility (conceptual)
    if planetary_compatibility.get('compatibility_flags'):
        has_major_conflict = False
        for flag in planetary_compatibility['compatibility_flags']:
            if "conflict" in flag.lower() or "caution" in flag.lower() or "problematic" in flag.lower():
                is_valid = False
                reasons.append(f"Astrological concern: {flag}")
                has_major_conflict = True
            elif "strong alignment" in flag.lower() or "amplified" in flag.lower() or "harmonizes well" in flag.lower():
                reasons.append(f"Astrological alignment: {flag}")
        if not has_major_conflict and not reasons: # If no conflicts and no positives, add a neutral note
             reasons.append("The suggested name's planetary influence shows no major conceptual astrological conflicts and some general alignment.")

    # Rule 5: Phonetic Vibration
    # Modified: Removed desired_outcome from call
    phonetic_analysis = get_phonetic_vibration_analysis(suggested_name)
    if not phonetic_analysis.get('is_harmonious'):
        is_valid = False
        reasons.append(f"Phonetic vibration concern: The name's sound may create subtle dissonance or not fully support the desired energy. {phonetic_analysis.get('qualitative_description', 'N/A')}")
    else:
        reasons.append(f"Phonetic vibration is harmonious: The name has a pleasant and supportive sound. {phonetic_analysis.get('qualitative_description', 'N/A')}")

    
    if is_valid and not reasons:
        reasons.append("The suggested name is numerologically sound and energetically balanced for general positive influence.")
    elif not is_valid and not reasons:
        reasons.append("The suggested name has fundamental numerological or astrological conflicts that make it unsuitable.")


    return is_valid, " ".join(reasons).strip()

# Function to add page numbers to the canvas (modified for onFirstPage/onLaterPages)
def _header_footer(canvas_obj, doc):
    canvas_obj.saveState()
    canvas_obj.setFont('Helvetica', 9)
    page_number_text = f"Page {doc.page}"
    canvas_obj.drawString(doc.rightMargin, 0.75 * inch, page_number_text)
    canvas_obj.restoreState()

# --- PDF Generation Function ---
def create_numerology_pdf(report_data: Dict) -> bytes:
    """
    Generates a PDF numerology report using ReportLab.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    
    profile_details = report_data.get('profile_details', {})
    confirmed_suggestions = report_data.get('confirmed_suggestions', [])

    # Custom styles for better readability and structure
    styles.add(ParagraphStyle(name='TitleStyle', fontSize=24, leading=28, alignment=TA_CENTER, spaceAfter=20, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SubHeadingStyle', fontSize=18, leading=22, spaceBefore=20, spaceAfter=10, alignment=TA_CENTER, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SectionHeadingStyle', fontSize=14, leading=18, spaceBefore=15, spaceAfter=8, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SubSectionHeadingStyle', fontSize=12, leading=16, spaceBefore=10, spaceAfter=4, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='BoldBodyText', fontSize=10, leading=14, spaceAfter=6, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='NormalBodyText', fontSize=10, leading=14, spaceAfter=6, fontName='Helvetica', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='BulletStyle', fontSize=10, leading=14, leftIndent=36, bulletIndent=18, spaceAfter=3, fontName='Helvetica'))
    styles.add(ParagraphStyle(name='ItalicBodyText', fontSize=10, leading=14, spaceAfter=6, fontName='Helvetica-Oblique', alignment=TA_JUSTIFY))
    
    # New styles for highlighting and aesthetics (slightly softer colors)
    styles.add(ParagraphStyle(name='ExecutiveSummaryStyle', fontSize=11, leading=16, spaceBefore=10, spaceAfter=12, fontName='Helvetica', alignment=TA_JUSTIFY,
                                backColor=HexColor('#E6F7FF'),
                                borderPadding=6, borderWidth=0.5, borderColor=HexColor('#A0D9EF'),
                                borderRadius=5))
    styles.add(ParagraphStyle(name='KeyInsightStyle', fontSize=10, leading=14, spaceBefore=8, spaceAfter=8, fontName='Helvetica-Bold', alignment=TA_JUSTIFY,
                                backColor=HexColor('#FFF8E1'),
                                borderPadding=4, borderWidth=0.5, borderColor=HexColor('#FFD700'),
                                borderRadius=3))
    styles.add(ParagraphStyle(name='CrucialTakeawayStyle', fontSize=10, leading=14, spaceBefore=8, spaceAfter=8, fontName='Helvetica-Bold', alignment=TA_JUSTIFY,
                                backColor=HexColor('#FCE4EC'),
                                borderPadding=4, borderWidth=0.5, borderColor=HexColor('#FFAB91'),
                                borderRadius=3))
    styles.add(ParagraphStyle(name='ImportantNoteStyle', fontSize=10, leading=14, spaceBefore=8, spaceAfter=8, fontName='Helvetica-Oblique', alignment=TA_JUSTIFY,
                                backColor=HexColor('#E0FFFF'),
                                borderPadding=4, borderWidth=0.5, borderColor=HexColor('#AFEEEE'),
                                borderRadius=3))
    styles.add(ParagraphStyle(name='QuoteStyle', fontSize=10, leading=14, spaceBefore=10, spaceAfter=10, leftIndent=20, rightIndent=20, fontName='Helvetica-Oblique', textColor=HexColor('#555555')))

    # Ensure standard styles are also configured for consistency
    styles['Heading1'].fontSize = 18
    styles['Heading1'].leading = 22
    styles['Heading1'].spaceBefore = 12
    styles['Heading1'].spaceAfter = 6
    styles['Heading1'].fontName = 'Helvetica-Bold'

    styles['Heading2'].fontSize = 14
    styles['Heading2'].leading = 18
    styles['Heading2'].spaceBefore = 10
    styles['Heading2'].spaceAfter = 4
    styles['Heading2'].fontName = 'Helvetica-Bold'

    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 14
    styles['BodyText'].spaceAfter = 6
    styles['BodyText'].alignment = TA_JUSTIFY


    styles['Bullet'].fontSize = 10
    styles['Bullet'].leading = 14
    styles['Bullet'].leftIndent = 36
    styles['Bullet'].bulletIndent = 18
    styles['Bullet'].spaceAfter = 3

    Story = []

    # Title Page / Header
    Story.append(Paragraph("🌟 Your Personalized Numerology Report 🌟", styles['TitleStyle']))
    Story.append(Spacer(1, 0.2 * inch))
    Story.append(Paragraph(f"For: <b>{report_data.get('full_name', 'Client Name')}</b>", styles['SubHeadingStyle']))
    Story.append(Paragraph(f"Birth Date: <b>{report_data.get('birth_date', 'N/A')}</b>", styles['SubSectionHeadingStyle']))
    if profile_details.get('birth_time'):
        Story.append(Paragraph(f"<b>Birth Time:</b> {profile_details.get('birth_time', 'N/A')}", styles['SubSectionHeadingStyle']))
    if profile_details.get('birth_place'):
        Story.append(Paragraph(f"<b>Birth Place:</b> {profile_details.get('birth_place', 'N/A')}", styles['SubSectionHeadingStyle']))
    Story.append(Spacer(1, 0.5 * inch))
    Story.append(Paragraph("A Comprehensive Guide to Your Energetic Blueprint and Name Optimization", styles['ItalicBodyText']))
    Story.append(PageBreak())

    # Main Report Content (from intro_response)
    main_report_content = report_data.get('intro_response', 'No report content available.')
    
    # Process Markdown content line by line to apply correct ReportLab styles
    lines = main_report_content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            Story.append(Spacer(1, 0.1 * inch))
            continue

        # Check for special highlighting patterns first
        if line.startswith('<b>Crucial Takeaway:</b>'):
            clean_line = line.replace('<b>Crucial Takeaway:</b>', '').strip()
            Story.append(Paragraph(f"<b>Crucial Takeaway:</b> {clean_line}", styles['CrucialTakeawayStyle']))
        elif line.startswith('<b>Key Insight:</b>'):
            clean_line = line.replace('<b>Key Insight:</b>', '').strip()
            Story.append(Paragraph(f"<b>Key Insight:</b> {clean_line}", styles['KeyInsightStyle']))
        elif line.startswith('<b>Important Note:</b>'):
            clean_line = line.replace('<b>Important Note:</b>', '').strip()
            Story.append(Paragraph(f"<b>Important Note:</b> {clean_line}", styles['ImportantNoteStyle']))
        elif line.startswith('### '):
            clean_line = line[4:].strip()
            clean_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_line)
            clean_line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', clean_line)
            Story.append(Paragraph(clean_line, styles['SubSectionHeadingStyle']))
        elif line.startswith('## '):
            clean_line = line[3:].strip()
            clean_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_line)
            clean_line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', clean_line)
            Story.append(Paragraph(clean_line, styles['SectionHeadingStyle']))
            Story.append(Spacer(1, 0.2 * inch))
            if "Executive Summary" not in clean_line and "Introduction" not in clean_line:
                 Story.append(PageBreak())
        elif line.startswith('* '):
            clean_line = line[2:].strip()
            clean_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_line)
            clean_line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', clean_line)
            Story.append(Paragraph(clean_line, styles['BulletStyle']))
        else:
            clean_line = line.strip()
            clean_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_line)
            clean_line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', clean_line)
            Story.append(Paragraph(clean_line, styles['NormalBodyText']))
    
    Story.append(PageBreak())

    # NEW: Confirmed Name Suggestions Section
    if confirmed_suggestions:
        Story.append(Paragraph("✅ Confirmed Name Corrections", styles['SubHeadingStyle']))
        Story.append(Paragraph("These are the name suggestions you have confirmed, along with their detailed numerological rationales.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.1 * inch))
        for suggestion in confirmed_suggestions:
            Story.append(Paragraph(f"<b>Name:</b> {suggestion.get('name', 'N/A')}", styles['BoldBodyText']))
            Story.append(Paragraph(f"<b>Expression Number:</b> {suggestion.get('expression_number', 'N/A')}", styles['NormalBodyText']))
            Story.append(Paragraph(f"<b>Rationale:</b> {suggestion.get('rationale', 'No rationale provided.')}", styles['NormalBodyText']))
            Story.append(Spacer(1, 0.2 * inch))
        Story.append(PageBreak())

    # NEW: Raw Data Sections are now consolidated and presented more cleanly
    Story.append(Paragraph("📊 Core Numerological Blueprint (Raw Data)", styles['SubHeadingStyle']))
    Story.append(Paragraph("This section provides the raw calculated data for your core numerological blueprint, which forms the foundation of the detailed interpretations found earlier in the report.", styles['NormalBodyText']))
    Story.append(Spacer(1, 0.1 * inch))

    Story.append(Paragraph(f"<b>Full Name:</b> {profile_details.get('full_name', 'N/A')}", styles['NormalBodyText']))
    Story.append(Paragraph(f"<b>Birth Date:</b> {profile_details.get('birth_date', 'N/A')}", styles['NormalBodyText']))
    if profile_details.get('birth_time'):
        Story.append(Paragraph(f"<b>Birth Time:</b> {profile_details.get('birth_time', 'N/A')}", styles['NormalBodyText']))
    if profile_details.get('birth_place'):
        Story.append(Paragraph(f"<b>Birth Place:</b> {profile_details.get('birth_place', 'N/A')}", styles['NormalBodyText']))
    # Removed: Desired Outcome from PDF raw data
    # Story.append(Paragraph(f"<b>Desired Outcome:</b> {profile_details.get('desired_outcome', 'N/A')}", styles['NormalBodyText']))
    Story.append(Spacer(1, 0.1 * inch))

    Story.append(Paragraph("<b>Expression Number:</b>", styles['BoldBodyText']))
    Story.append(Paragraph(f"Value: {profile_details.get('expression_number', 'N/A')} (Ruled by {profile_details.get('expression_details', {}).get('planetary_ruler', 'N/A')})", styles['NormalBodyText']))
    Story.append(Paragraph(f"Total before reduction: {profile_details.get('expression_details', {}).get('total_before_reduction', 'N/A')}", styles['NormalBodyText']))
    Story.append(Paragraph(f"Letter Breakdown: {json.dumps(profile_details.get('expression_details', {}).get('letter_breakdown', {}))}", styles['NormalBodyText']))
    Story.append(Spacer(1, 0.1 * inch))

    Story.append(Paragraph("<b>Life Path Number:</b>", styles['BoldBodyText']))
    Story.append(Paragraph(f"Value: {profile_details.get('life_path_number', 'N/A')}", styles['NormalBodyText']))
    Story.append(Paragraph(f"Details: Original Month {profile_details.get('life_path_details', {}).get('original_month', 'N/A')}, Original Day {profile_details.get('life_path_details', {}).get('original_day', 'N/A')}, Original Year {profile_details.get('life_path_details', {}).get('original_year', 'N/A')}", styles['NormalBodyText']))
    Story.append(Paragraph(f"Total sum of all digits before final reduction: {profile_details.get('life_path_details', {}).get('total_sum_all_digits_before_reduction', 'N/A')}", styles['NormalBodyText']))
    if profile_details.get('life_path_details', {}).get('karmic_debt_in_components'):
        Story.append(Paragraph(f"Karmic Debt in Birth Date Components: {', '.join(map(str, profile_details.get('life_path_details', {}).get('karmic_debt_in_components')))}", styles['NormalBodyText']))
    Story.append(Spacer(1, 0.1 * inch))

    Story.append(Paragraph("<b>Birth Day Number:</b>", styles['BoldBodyText']))
    Story.append(Paragraph(f"Value: {profile_details.get('birth_day_number', 'N/A')}", styles['NormalBodyText']))
    Story.append(Paragraph(f"Original Day: {profile_details.get('birth_day_details', {}).get('original_day', 'N/A')}", styles['NormalBodyText']))
    Story.append(Spacer(1, 0.1 * inch))

    Story.append(Paragraph("<b>Soul Urge Number:</b>", styles['BoldBodyText']))
    Story.append(Paragraph(f"Value: {profile_details.get('soul_urge_number', 'N/A')}", styles['NormalBodyText']))
    Story.append(Paragraph(f"Total before reduction: {profile_details.get('soul_urge_details', {}).get('total_before_reduction', 'N/A')}", styles['NormalBodyText']))
    Story.append(Spacer(1, 0.1 * inch))

    Story.append(Paragraph("<b>Personality Number:</b>", styles['BoldBodyText']))
    Story.append(Paragraph(f"Value: {profile_details.get('personality_number', 'N/A')}", styles['NormalBodyText']))
    Story.append(Paragraph(f"Total before reduction: {profile_details.get('personality_details', {}).get('total_before_reduction', 'N/A')}", styles['NormalBodyText']))
    Story.append(Spacer(1, 0.2 * inch))
    Story.append(PageBreak())


    # Lo Shu Grid Analysis (Raw Data)
    if profile_details.get('lo_shu_grid'):
        lo_shu = profile_details['lo_shu_grid']
        Story.append(Paragraph("🗺️ Lo Shu Grid Analysis (Raw Data)", styles['SubHeadingStyle']))
        Story.append(Paragraph("This section provides the raw digit counts from your birth date, forming your unique Lo Shu Grid pattern.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.1 * inch))

        Story.append(Paragraph(f"<b>Digits in Birth Date</b>: {json.dumps(lo_shu.get('grid_counts'))}", styles['NormalBodyText']))
        if lo_shu.get('missing_numbers'):
            Story.append(Paragraph("<b>Missing Numbers</b>:", styles['BoldBodyText']))
            for num_data in lo_shu['missing_lessons']: # Use missing_lessons for detailed impact
                Story.append(Paragraph(f"• {num_data['number']}: {num_data['impact']}", styles['BulletStyle']))
        else:
            Story.append(Paragraph("All numbers 1-9 are present in your birth date, indicating a balanced Lo Shu Grid.", styles['NormalBodyText']))
        
        if lo_shu.get('grid_updated_by_name'):
             Story.append(Paragraph(f"<i>Note: This grid includes the conceptual influence of the suggested name's Expression Number.</i>", styles['ItalicBodyText']))
        Story.append(Spacer(1, 0.2 * inch))
        Story.append(PageBreak())

    # Astrological Integration (Conceptual Raw Data)
    if profile_details.get('astro_info'):
        astro = profile_details['astro_info']
        Story.append(Paragraph("🌌 Astro-Numerological Insights (Conceptual Raw Data)", styles['SubHeadingStyle']))
        Story.append(Paragraph("This section provides the conceptual astrological data used for integrated analysis. These are simplified representations for numerological context, not a full astrological chart.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.1 * inch))

        Story.append(Paragraph(f"<b>Ascendant/Lagna (Rising Sign)</b>: {astro.get('ascendant_info', {}).get('sign', 'N/A')} (Conceptual Ruler: {astro.get('ascendant_info', {}).get('ruler', 'N/A')})", styles['NormalBodyText']))
        Story.append(Paragraph(f"<b>Notes:</b> {astro.get('ascendant_info', {}).get('notes', 'N/A')}", styles['ItalicBodyText']))
        Story.append(Spacer(1, 0.1 * inch))

        Story.append(Paragraph(f"<b>Moon Sign/Rashi</b>: {astro.get('moon_sign_info', {}).get('sign', 'N/A')} (Conceptual Ruler: {astro.get('moon_sign_info', {}).get('ruler', 'N/A')})", styles['NormalBodyText']))
        Story.append(Paragraph(f"<b>Notes:</b> {astro.get('moon_sign_info', {}).get('notes', 'N/A')}", styles['ItalicBodyText']))
        Story.append(Spacer(1, 0.1 * inch))
        
        Story.append(Paragraph("<b>Conceptual Planetary Lords & Degrees (Benefic/Malefic Influences)</b>:", styles['BoldBodyText']))
        if astro.get('planetary_lords'):
            for lord in astro['planetary_lords']:
                degree_info = f" ({lord['degree']})" if 'degree' in lord and lord['degree'] != 'N/A' else ""
                Story.append(Paragraph(f"• {lord['planet']} ({lord['nature']}){degree_info}", styles['BulletStyle']))
        else:
            Story.append(Paragraph("No specific conceptual planetary lords identified.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.1 * inch))

        if astro.get('planetary_compatibility', {}).get('compatibility_flags'):
            Story.append(Paragraph("<b>Planetary Compatibility Flags for Suggested Name (Expression Number)</b>:", styles['BoldBodyText']))
            for flag in astro['planetary_compatibility']['compatibility_flags']:
                Story.append(Paragraph(f"• {flag}", styles['BulletStyle']))
        else:
            Story.append(Paragraph("No specific planetary compatibility flags identified for the suggested name based on conceptual analysis.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.2 * inch))
        Story.append(PageBreak())

    # Phonetic Vibration (Conceptual Raw Data)
    if profile_details.get('phonetic_vibration'):
        phonetics = profile_details['phonetic_vibration']
        Story.append(Paragraph("🗣️ Phonetic Vibration Analysis (Conceptual Raw Data)", styles['SubHeadingStyle']))
        Story.append(Paragraph("This section provides the raw data from the conceptual phonetic analysis of your name.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.1 * inch))

        Story.append(Paragraph(f"<b>Overall Harmony Score</b>: {phonetics['score']:.2f} ({'Harmonious' if phonetics['is_harmonious'] else 'Needs Consideration'})", styles['NormalBodyText']))
        Story.append(Paragraph(f"<b>Qualitative Description</b>: {phonetics.get('qualitative_description', 'N/A')}", styles['NormalBodyText']))
        if phonetics.get('notes'):
            Story.append(Paragraph("<b>Notes on Sound Vibration</b>:", styles['BoldBodyText']))
            for note in phonetics['notes']:
                Story.append(Paragraph(f"• {note}", styles['BulletStyle']))
        Story.append(Spacer(1, 0.2 * inch))
        Story.append(PageBreak())

    # Edge Case Handling (Raw Data)
    if profile_details.get('edge_cases'):
        edge_cases_data = profile_details['edge_cases']
        if edge_cases_data:
            Story.append(Paragraph("⚠️ Identified Edge Cases & Special Considerations (Raw Data)", styles['SubHeadingStyle']))
            Story.append(Paragraph("This section lists the raw data for identified edge cases in your numerological profile.", styles['NormalBodyText']))
            Story.append(Spacer(1, 0.1 * inch))
            for ec in edge_cases_data:
                Story.append(Paragraph(f"<b>Type</b>: {ec['type']}", styles['BoldBodyText']))
                Story.append(Paragraph(f"Description: {ec['description']}", styles['NormalBodyText']))
                Story.append(Paragraph(f"Resolution Guidance: {ec['resolution_guidance']}", styles['NormalBodyText']))
                Story.append(Spacer(1, 0.1 * inch))
        Story.append(Spacer(1, 0.2 * inch))
        Story.append(PageBreak())

    # Temporal Numerology (Raw Data)
    if profile_details.get('timing_recommendations'):
        timing = profile_details['timing_recommendations']
        Story.append(Paragraph("⏰ Temporal Numerology (Raw Data)", styles['SubHeadingStyle']))
        Story.append(Paragraph("This section provides the raw data for your current and future numerological cycles.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.1 * inch))

        Story.append(Paragraph(f"<b>Current Personal Year</b>: {timing['current_personal_year']}", styles['NormalBodyText']))
        Story.append(Paragraph(f"<b>Optimal Activities</b>: {', '.join(timing['optimal_activities'])}", styles['NormalBodyText']))
        Story.append(Paragraph(f"<b>Best Months for Action/Implementation</b>: {', '.join(map(str, timing['best_months_for_action']))}", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.2 * inch))
        Story.append(PageBreak())

    # Shadow Work & Growth Edge (Raw Data)
    if profile_details.get('potential_challenges', {}).get('potential_challenges'):
        challenges = profile_details['potential_challenges']
        Story.append(Paragraph("🚧 Shadow Work & Growth Edge (Raw Data)", styles['SubHeadingStyle']))
        Story.append(Paragraph("This section contains the raw data for potential challenges and growth areas.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.1 * inch))

        Story.append(Paragraph("<b>Potential Challenges Identified</b>:", styles['BoldBodyText']))
        for challenge in challenges['potential_challenges']:
            Story.append(Paragraph(f"• {challenge}", styles['BulletStyle']))
        Story.append(Paragraph("<b>Mitigation Strategies</b>:", styles['BoldBodyText']))
        for strategy in challenges['mitigation_strategies']:
            Story.append(Paragraph(f"• {strategy}", styles['BulletStyle']))
        Story.append(Paragraph("<b>Growth Opportunities</b>:", styles['BoldBodyText']))
        for opportunity in challenges['growth_opportunities']:
            Story.append(Paragraph(f"• {opportunity}", styles['BulletStyle']))
        Story.append(Spacer(1, 0.2 * inch))
        Story.append(PageBreak())

    # Personalized Development Blueprint (Raw Data)
    if profile_details.get('development_recommendations'):
        dev_recs = profile_details['development_recommendations']
        Story.append(Paragraph("🌱 Personalized Development Blueprint (Raw Data)", styles['SubHeadingStyle']))
        Story.append(Paragraph("This section provides the raw data for your personalized development recommendations.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.1 * inch))

        Story.append(Paragraph(f"<b>Immediate Focus</b>: {dev_recs['immediate_focus']}", styles['NormalBodyText']))
        Story.append(Paragraph(f"<b>Long-Term Goal</b>: {dev_recs['long_term_goal']}", styles['NormalBodyText']))
        if dev_recs['karmic_work']:
            Story.append(Paragraph(f"<b>Key Karmic Work Areas</b>: {', '.join(dev_recs['karmic_work'])}", styles['NormalBodyText']))
        Story.append(Paragraph("<b>Recommended Monthly Practices</b>:", styles['BoldBodyText']))
        for practice in dev_recs['monthly_practices']:
            Story.append(Paragraph(f"• {practice}", styles['BulletStyle']))
        Story.append(Spacer(1, 0.2 * inch))
        Story.append(PageBreak())

    # Future Cycles Forecast (Raw Data)
    if profile_details.get('yearly_forecast'):
        Story.append(Paragraph("🗓️ Future Cycles Forecast (Raw Data)", styles['SubHeadingStyle']))
        Story.append(Paragraph("This section provides the raw data for your multi-year numerological forecast.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.1 * inch))
        for year, forecast_data in profile_details['yearly_forecast'].items():
            Story.append(Paragraph(f"<b>Year {year} (Personal Year {forecast_data['personal_year']})</b>:", styles['BoldBodyText']))
            Story.append(Paragraph(f"Theme: {forecast_data['theme']}", styles['NormalBodyText']))
            Story.append(Paragraph(f"Focus: {forecast_data['focus_areas']}", styles['NormalBodyText']))
            Story.append(Paragraph(f"Energy Level: {forecast_data['energy_level']}", styles['NormalBodyText']))
            Story.append(Paragraph(f"Optimal Months: {', '.join(map(str, forecast_data['optimal_months']))}", styles['NormalBodyText']))
            Story.append(Spacer(1, 0.1 * inch))
        Story.append(Spacer(1, 0.2 * inch))
        Story.append(PageBreak())

    # Numerical Uniqueness Profile (Raw Data)
    if profile_details.get('uniqueness_score'):
        uniqueness = profile_details['uniqueness_score']
        Story.append(Paragraph("✨ Numerical Uniqueness Profile (Raw Data)", styles['SubHeadingStyle']))
        Story.append(Paragraph("This section presents the raw data for your numerical uniqueness assessment.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.1 * inch))

        Story.append(Paragraph(f"<b>Your Profile Uniqueness Score</b>: {uniqueness['score']:.2f} ({uniqueness['interpretation']})", styles['NormalBodyText']))
        Story.append(Paragraph(f"<b>Key Rarity Factors</b>: {', '.join(uniqueness['rarity_factors'])}", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.2 * inch))
        Story.append(PageBreak())

    # Build the PDF with page numbering
    doc.build(Story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    buffer.seek(0)
    return buffer.getvalue()

# --- Name Suggestion Engine Class ---
class NameSuggestionEngine:
    @staticmethod
    # Modified: Removed desired_outcome parameter
    def determine_target_numbers_for_outcome() -> List[int]:
        """
        Determines optimal Expression Numbers for general well-being and balance.
        This is a rule-based mapping, now without 'desired_outcome' input.
        """
        # Default set of generally positive/balanced numbers
        return sorted(list(set([1, 3, 5, 6, 8, 9, 11, 22, 33])))

    @staticmethod
    # Modified: Removed desired_outcome parameter
    async def generate_name_suggestions(llm_instance: ChatGoogleGenerativeAI, original_full_name: str, target_expression_numbers: List[int]) -> NameSuggestionsOutput:
        """
        Generates name suggestions using the LLM and then recalculates Expression Numbers
        using the backend's numerology logic for accuracy.
        """
        # The PydanticOutputParser defines the desired schema.
        pydantic_parser = PydanticOutputParser(pydantic_object=NameSuggestionsOutput)

        # NEW: The OutputFixingParser wraps the original parser and uses an LLM to correct parsing errors.
        output_fixing_parser = OutputFixingParser.from_llm(parser=pydantic_parser, llm=llm_instance)
        
        parser_instructions = output_fixing_parser.get_format_instructions()
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=NAME_SUGGESTION_SYSTEM_PROMPT.format(parser_instructions=parser_instructions)),
            HumanMessage(content=NAME_SUGGESTION_HUMAN_PROMPT.format(
                original_full_name=original_full_name,
                # Removed: desired_outcome from prompt
                target_expression_numbers=target_expression_numbers
            ))
        ])
        
        # The chain now uses the robust OutputFixingParser.
        chain = prompt | llm_instance | output_fixing_parser
        
        try:
            logger.info("Generating name suggestions with OutputFixingParser.")
            llm_response_output = await chain.ainvoke({})
            
            # --- CRITICAL: Recalculate Expression Numbers for LLM suggestions ---
            # This ensures the numerological feasibility is based on our Python logic, not just LLM's text generation.
            # The OutputFixingParser should have already corrected the output to be valid JSON,
            # so each suggestion.expression_number should be a valid integer.
            for suggestion in llm_response_output.suggestions:
                calculated_exp_num, _ = calculate_expression_number_with_details(suggestion.name)
                # We overwrite the LLM's calculation with our own verified one.
                suggestion.expression_number = calculated_exp_num
            # --- END CRITICAL SECTION ---

            logger.info("Successfully generated and validated name suggestions.")
            return llm_response_output
        except Exception as e:
            # If even the OutputFixingParser fails, we log the error and raise a clear exception.
            logger.error(f"LLM Name Suggestion Generation Error even after attempting to fix: {e}", exc_info=True)
            raise ValueError(f"Failed to generate name suggestions: {e}")

# --- Flask Routes ---
@app.route('/')
def home():
    """Basic home route for health check."""
    return "Hello from Flask!"
@app.route('/initial_suggestions', methods=['POST'])
@limiter.limit("10 per minute")
async def initial_suggestions_endpoint():
    data = request.json
    full_name = data.get('full_name')
    birth_date = data.get('birth_date')

    if not all([full_name, birth_date]):
        return jsonify({"error": "Missing full_name or birth_date for initial suggestions."}), 400

    try:
        profile_data = await get_comprehensive_numerology_profile(
            full_name=full_name,
            birth_date=birth_date,
            birth_time=data.get('birth_time'),
            birth_place=data.get('birth_place'),
        )

        target_numbers = NameSuggestionEngine.determine_target_numbers_for_outcome()

        name_suggestions_output = await NameSuggestionEngine.generate_name_suggestions(
            llm_manager.creative_llm,
            full_name,
            target_numbers
        )

        logger.info(f"Generated Initial Name Suggestions: {name_suggestions_output.json()}")

        suggestions_data = name_suggestions_output.dict()

        # STRICT VALIDATION
        life_path_number = profile_data.get('life_path_number')
        birth_day_number = profile_data.get('birth_day_number')

        filtered = filter_strictly_valid_suggestions(
            suggestions_data["suggestions"],
            life_path_number,
            birth_day_number
        )

# ✅ Wrap in expected object format
        suggestions_data["suggestions"] = [
        suggestion for suggestion in suggestions_data["suggestions"]
        if suggestion["name"] in filtered
]


        return jsonify({
                        "suggestions": suggestions_data["suggestions"],
                        "reasoning": suggestions_data["reasoning"],
                        "profile_data": profile_data
                    }), 200

    except Exception as e:
        logger.error(f"Error generating initial suggestions: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred while generating initial suggestions. Please try again later."}), 500



@app.route('/validate_name', methods=['POST'])
@limiter.limit("30 per minute") # Apply limiter directly
async def validate_name_endpoint():
    """
    Validates a single suggested name against the client's profile and rules.
    Returns a clear YES/NO and a detailed rationale.
    """
    data = request.json
    suggested_name = data.get('suggested_name')
    client_profile = data.get('client_profile')

    # --- FIX START: Explicit Input Validation for /validate_name ---
    if not data:
        logger.error("Validation request received with no JSON data.")
        return jsonify({"error": "No data provided for validation."}), 400

    if not suggested_name or not isinstance(suggested_name, str) or not suggested_name.strip():
        logger.error(f"Validation request missing or invalid 'suggested_name': '{suggested_name}'")
        return jsonify({"error": "Missing or empty 'suggested_name' for validation."}), 400

    if not client_profile or not isinstance(client_profile, dict):
        logger.error(f"Validation request missing or invalid 'client_profile': '{client_profile}'")
        return jsonify({"error": "Missing or invalid 'client_profile' for validation."}), 400
    # --- FIX END: Explicit Input Validation for /validate_name ---

    try:
        # validate_suggested_name_rules is synchronous, so no await needed here
        # Modified: Removed desired_outcome from call
        is_valid, rationale = validate_suggested_name_rules(suggested_name, client_profile)
        
        # Calculate expression number for the suggested name to return it
        suggested_exp_num, _ = calculate_expression_number_with_details(suggested_name)
        
        # Calculate other live values for the suggested name
        first_name_value = calculate_single_digit(sum(get_chaldean_value(char) for char in clean_name(suggested_name).split(' ')[0]), False)
        soul_urge_num, _ = calculate_soul_urge_number_with_details(suggested_name)
        personality_num, _ = calculate_personality_number_with_details(suggested_name)
    
        karmic_debt_present = bool(check_karmic_debt(suggested_name))


        return jsonify({
            "suggested_name": suggested_name,
            "is_valid": is_valid,
            "rationale": rationale,
            "expression_number": suggested_exp_num, # Ensure this is the calculated value
            "first_name_value": first_name_value,
            "soul_urge_number": soul_urge_num,
            "personality_number": personality_num,
            "karmic_debt_present": karmic_debt_present
        }), 200

    except Exception as e:
        logger.error(f"Error validating name: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during name validation. Please try again later."}), 500

@app.route('/generate_pdf_report', methods=['POST'])
@limiter.limit("5 per hour") # Apply limiter directly
async def generate_pdf_report_endpoint():
    """
    Endpoint to generate and return a PDF numerology report.
    """
    report_data_from_frontend = request.json
    
    # Removed desired_outcome from validation
    if not all([report_data_from_frontend.get('full_name'), report_data_from_frontend.get('birth_date'), report_data_from_frontend.get('confirmed_suggestions') is not None]):
        return jsonify({"error": "Missing essential data (full_name, birth_date, or confirmed_suggestions) for PDF generation."}), 400

    try:
        profile_details = await get_comprehensive_numerology_profile(
            full_name=report_data_from_frontend['full_name'],
            birth_date=report_data_from_frontend['birth_date'],
            birth_time=report_data_from_frontend.get('birth_time'),
            birth_place=report_data_from_frontend.get('birth_place'),
        )

        # 🛡️ Hardened logic
        profile_details['confirmed_suggestions_raw'] = report_data_from_frontend.get('confirmed_suggestions', [])

        filtered_confirmed = filter_strictly_valid_suggestions(
        profile_details['confirmed_suggestions_raw'],
        profile_details.get('life_path_number'),
        profile_details.get('birth_day_number')
        )
        profile_details['confirmed_suggestions'] = filtered_confirmed


        llm_input_data = json.dumps(profile_details, indent=2)
        report_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=ADVANCED_REPORT_SYSTEM_PROMPT),
            HumanMessage(content=ADVANCED_REPORT_HUMAN_PROMPT.format(llm_input_data=llm_input_data))
        ])
        
        report_chain = report_prompt | llm_manager.creative_llm
        
        logger.info("Calling LLM for advanced report generation...")
        llm_report_response = await report_chain.ainvoke({})
        logger.info("LLM advanced report generation complete.")

        pdf_data = {
            "full_name": profile_details['full_name'],
            "birth_date": profile_details['birth_date'],
            "profile_details": profile_details,
            "intro_response": llm_report_response.content,
            "confirmed_suggestions": profile_details['confirmed_suggestions']
        }

        pdf_bytes = create_numerology_pdf(pdf_data)
        
        logger.info(f"Generated PDF bytes size: {len(pdf_bytes)} bytes")

        filename = f"Numerology_Report_{report_data_from_frontend['full_name'].replace(' ', '_')}.pdf"
        
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error(f"Error generating PDF report: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate PDF report: {e}"}), 500

@app.route('/generate_text_report', methods=['POST'])
@limiter.limit("5 per minute") # Apply limiter directly
async def generate_text_report_endpoint():
    """
    Endpoint to generate and return the numerology report content as plain text (Markdown).
    This is used for the frontend preview.
    """
    report_data_from_frontend = request.json
    
    # Removed desired_outcome from validation
    if not all([report_data_from_frontend.get('full_name'), report_data_from_frontend.get('birth_date'), report_data_from_frontend.get('confirmed_suggestions') is not None]):
        return jsonify({"error": "Missing essential data (full_name, birth_date, or confirmed_suggestions) for text report generation."}), 400

    try:
        # Modified: Removed desired_outcome from call
        # Calculate profile
        profile_details = await get_comprehensive_numerology_profile(
        full_name=report_data_from_frontend['full_name'],
        birth_date=report_data_from_frontend['birth_date'],
        birth_time=report_data_from_frontend.get('birth_time'),
        birth_place=report_data_from_frontend.get('birth_place'),)

# Save raw (optional)
        profile_details['confirmed_suggestions_raw'] = report_data_from_frontend.get('confirmed_suggestions', [])

# ✅ Re-validate names before using in LLM
        filtered_confirmed = filter_strictly_valid_suggestions(
        profile_details['confirmed_suggestions_raw'],
        profile_details.get('life_path_number'),
        profile_details.get('birth_day_number'))
        profile_details['confirmed_suggestions'] = filtered_confirmed

        llm_input_data = json.dumps(profile_details, indent=2)
        
        report_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=ADVANCED_REPORT_SYSTEM_PROMPT),
            HumanMessage(content=ADVANCED_REPORT_HUMAN_PROMPT.format(llm_input_data=llm_input_data))
        ])
        
        report_chain = report_prompt | llm_manager.creative_llm
        
        logger.info("Calling LLM for text report generation (preview)...")
        llm_report_response = await report_chain.ainvoke({})
        logger.info("LLM text report generation complete.")

        return jsonify({"report_content": llm_report_response.content}), 200

    except Exception as e:
        logger.error(f"Error generating text report for preview: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate text report for preview: {e}"}), 500



# Error handlers
@app.errorhandler(400)
def bad_request(error):
    logger.error(f"Bad Request: {error}")
    return jsonify({"error": "Bad Request: " + str(error.description)}), 400

@app.errorhandler(404)
def not_found(error):
    logger.error(f"Not Found: {error}")
    return jsonify({"error": "Not Found: The requested URL was not found on the server."}), 404

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal Server Error: {error}", exc_info=True)
    return jsonify({"error": "Internal Server Error: The server encountered an internal error and was unable to complete your request. Please try again later."}), 500

# This is the WSGI application that Uvicorn will serve.
# It wraps your Flask 'app' instance.
asgi_app = WSGIMiddleware(app)

if __name__ == '__main__':
    # For local testing, you can run this file directly with `python app.py`
    # or use uvicorn to test the ASGI compatibility:
    # uvicorn app:asgi_app --host 0.0.0.0 --port 8000
    app.run(debug=True, host='0.0.0.0', port=os.getenv('PORT', 5000))
# END OF app.py - DO NOT DELETE THIS LINE
